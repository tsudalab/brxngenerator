"""
Provides `DebugMode`, an evaluation mode for debugging theano internals.

TODO: add support for IfElse Op, LazyLinker, PureOp, etc.

"""
from __future__ import absolute_import, print_function, division

import copy
import sys
import gc
import logging
import six.moves.copyreg as copyreg
from itertools import chain, product as itertools_product
from theano.compat import izip

import numpy

import theano
from theano import gof, config
from theano.compat import get_unbound_function
from six import iteritems, itervalues
from six.moves import StringIO, xrange
from theano.gof import (graph, utils, link, ops_with_inner_function)
from theano.gof.link import raise_with_op
from theano.compile.function_module import (
    FunctionMaker, Function, infer_reuse_pattern,
    SymbolicInputKit, SymbolicOutput, Supervisor, std_fgraph)
from theano.compile.mode import Mode, register_mode
from theano.compile.ops import OutputGuard

__docformat__ = "restructuredtext en"
_logger = logging.getLogger("theano.compile.debugmode")


class NoDuplicateOptWarningFilter(logging.Filter):
    prev_msgs = set([])

    def filter(self, record):
        msg = record.getMessage()
        if msg.startswith('Optimization Warning: '):
            if msg in self.prev_msgs:
                return False
            else:
                self.prev_msgs.add(msg)
                return True
        return True

_logger.addFilter(NoDuplicateOptWarningFilter())


class DebugModeError(Exception):
    """
    Generic Exception raised to indicate an internal theano problem.

    """

    pass


class BadThunkOutput(DebugModeError):
    """
    Exception: Calling the same Op twice gives inconsistent outputs.

    It can be raised, for instance, if an Op's c_code and perform method
    do not agree, or if one of these methods do not give the same result
    when called twice with the same inputs (but different memory layouts
    for the output).

    """

    r = None
    """
    The `Variable` instance for which conflicting values were computed.

    """

    thunk1 = ''
    val1 = None
    """
    The value computed by `thunk1`.

    """

    thunk2 = ''
    val2 = None
    """
    The value computed by `thunk2`.

    """

    def __init__(self, r, thunk1, val1, thunk2, val2, inputs_val=()):
        super(BadThunkOutput, self).__init__()
        self.r = r
        self.thunk1 = thunk1
        self.val1 = val1
        self.thunk2 = thunk2
        self.val2 = val2
        self.inputs_val = inputs_val

    def offending_op(self):
        """
        Return the Op class whose c_code and perform implementations
        didn't match.

        """
        return type(self.r.owner.op)

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """
        Return a pretty multiline string representing the cause of
        the exception.

        """
        sio = StringIO()
        print("BadThunkOutput", file=sio)
        print("  Apply   :", self.r.owner, file=sio)
        print("  op      :", self.offending_op(), file=sio)
        print("  Outputs Type:", self.r.type, file=sio)
        print("  Outputs Shape:", getattr(self.val1, 'shape', None), file=sio)
        print("  Outputs Strides:", getattr(self.val1, 'strides', None),
              file=sio)
        print("  Inputs Type :", [i.type for i in self.r.owner.inputs],
              file=sio)
        print("  Inputs Shape:", [getattr(val, 'shape', None)
                                  for val in self.inputs_val], file=sio)
        print("  Inputs Strides:", [getattr(val, 'strides', None)
                                    for val in self.inputs_val], file=sio)
        scalar_values = []
        for ipt in self.inputs_val:
            if getattr(ipt, "size", -1) <= 10:
                scalar_values.append(ipt)
            else:
                scalar_values.append("not shown")
        print("  Inputs values: %s" % scalar_values, file=sio)
        print("  Bad Variable:", self.r, file=sio)
        print("  thunk1  :", self.thunk1, file=sio)
        print("  thunk2  :", self.thunk2, file=sio)

        utt = theano.tests.unittest_tools
        print(utt.str_diagnostic(self.val1, self.val2, None, None), file=sio)
        ret = sio.getvalue()
        return ret


class BadOptimization(DebugModeError):
    """
    Exception: some variable and its substitute take different runtime values.

    """

    new_r = None
    """
    A `Variable` instance that took a different value from `old_r`,
    but which replaced `old_r`.

    """

    old_r = None
    """
    A `Variable` instance that was replaced by `new_r`.

    """

    old_r_val = None
    """
    The value computed for `old_r`.

    """

    new_r_val = None
    """
    The value computed for `new_r`.

    """

    reason = None
    """
    An object that indicates why old_r was turned into new_r.

    Convention is that this is the name of the optimization that
    requested the replacement.

    """

    old_graph = ""
    """
    A multiline string representation of the graph leading to
    old_r, at the time of the replacement.

    """

    new_graph = ""
    """
    A multiline string representation of the graph leading to
    new_r, at the time of the replacement.

    """

    def __init__(self, old_r, new_r, old_r_val, new_r_val, reason,
                 old_graph, new_graph):
        super(BadOptimization, self).__init__()
        self.old_r = old_r
        self.new_r = new_r
        self.old_r_val = old_r_val
        self.new_r_val = new_r_val
        self.reason = reason
        self.old_graph = old_graph
        self.new_graph = new_graph

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """
        Return a pretty multiline string representating the cause
        of the exception.

        """
        sio = StringIO()
        val_str_len_limit = 800
        print("BadOptimization Error", super(BadOptimization,
                                             self).__str__(), file=sio)
        print("  Variable: id", id(self.new_r), self.new_r, file=sio)
        print("  Op", self.new_r.owner, file=sio)
        print("  Value Type:", type(self.new_r_val), file=sio)
        try:
            ssio = StringIO()
            print("  Old Value shape, dtype, strides:", end=' ', file=ssio)
            print(self.old_r_val.shape, end=' ', file=ssio)
            print(self.old_r_val.dtype, end=' ', file=ssio)
            print(self.old_r_val.strides, file=ssio)
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass

        str_old_r_val = str(self.old_r_val)
        if len(str_old_r_val) > val_str_len_limit:
            print("  Old Value: ", str(self.old_r_val)[
                :val_str_len_limit], '...', file=sio)
        else:
            print("  Old Value: ", str(self.old_r_val), file=sio)

        try:
            ssio = StringIO()
            print("  New Value shape, dtype, strides:", end=' ', file=ssio)
            print(self.new_r_val.shape, end=' ', file=ssio)
            print(self.new_r_val.dtype, end=' ', file=ssio)
            print(self.new_r_val.strides, file=ssio)
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass
        str_new_r_val = str(self.new_r_val)
        if len(str_new_r_val) > val_str_len_limit:
            print("  New Value: ", str(self.new_r_val)[
                :val_str_len_limit], '...', file=sio)
        else:
            print("  New Value: ", str(self.new_r_val), file=sio)

        try:
            ov = numpy.asarray(self.old_r_val)
            nv = numpy.asarray(self.new_r_val)
            ssio = StringIO()
            abs_diff = numpy.absolute(nv - ov)
            print("  Max Abs Diff: ", numpy.max(abs_diff), file=ssio)
            print("  Mean Abs Diff: ", numpy.mean(abs_diff), file=ssio)
            print("  Median Abs Diff: ", numpy.median(abs_diff), file=ssio)
            print("  Std Abs Diff: ", numpy.std(abs_diff), file=ssio)
            arg_max_val = numpy.argmax(abs_diff)
            values_at_max = (nv.flatten()[arg_max_val],
                             ov.flatten()[arg_max_val])
            print("  Value at Max Diff: ", values_at_max, file=ssio)

            reldiff = (abs_diff /
                       numpy.maaximum(numpy.absolute(nv) + numpy.absolute(ov),
                                      1e-8))
            print("  Max Rel Diff: ", numpy.max(reldiff), file=ssio)
            print("  Mean Rel Diff: ", numpy.mean(reldiff), file=ssio)
            print("  Median Rel Diff: ", numpy.median(reldiff), file=ssio)
            print("  Std Rel Diff: ", numpy.std(reldiff), file=ssio)
            arg_max_val = numpy.argmax(reldiff)
            values_at_max = (nv.flatten()[arg_max_val],
                             ov.flatten()[arg_max_val])
            print("  Value at Max Diff: ", values_at_max, file=ssio)
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass

        print("  Reason: ", str(self.reason), file=sio)
        print("  Old Graph:", file=sio)
        print(self.old_graph, file=sio)
        print("  New Graph:", file=sio)
        print(self.new_graph, file=sio)
        print("", file=sio)
        print("Hint: relax the tolerance by setting tensor.cmp_sloppy=1",
              file=sio)
        print("  or even tensor.cmp_sloppy=2 for less-strict comparison",
              file=sio)
        return sio.getvalue()


class BadDestroyMap(DebugModeError):
    """
    Exception: Some perform() or c_code() modified an input that
    wasn't in the destroy_map.

    """
    def __init__(self, node, idx, old_val, new_val, perform):
        super(BadDestroyMap, self).__init__()
        self.node = node
        self.idx = idx
        self.old_val = old_val
        self.new_val = new_val
        self.perform = perform

    def __str__(self):
        sio = StringIO()
        print("  node:", self.node, file=sio)
        print("  perform:", self.perform, file=sio)
        print("  node.inputs:", [(str(i), id(i))
                                 for i in self.node.inputs], file=sio)
        print("  destroy_map:", getattr(self.node.op,
                                        'destroy_map', {}), file=sio)
        print("  changed input idx:", self.idx, file=sio)
        print("  changed input type:", self.node.inputs[self.idx].type,
              file=sio)
        print("  repr (old val):", repr(self.old_val), file=sio)
        print("  repr (new val):", repr(self.new_val), file=sio)
        try:
            npy_old_val = numpy.asarray(self.old_val)
            npy_new_val = numpy.asarray(self.new_val)
            print("  value dtype (new <space> old):", npy_new_val.dtype,
                  npy_old_val.dtype, file=sio)
            print("  value shape (new <space> old):", npy_new_val.shape,
                  npy_old_val.shape, file=sio)
            print("  value min (new <space> old):", npy_new_val.min(),
                  npy_old_val.min(), file=sio)
            print("  value max (new <space> old):", npy_new_val.max(),
                  npy_old_val.max(), file=sio)
            delta = npy_new_val - npy_old_val
            print("  value min (new-old):", delta.min(), file=sio)
            print("  value max (new-old):", delta.max(), file=sio)
            print("  value argmin (new-old):",
                  numpy.unravel_index(delta.argmin(), npy_new_val.shape),
                  file=sio)
            print("  value argmax (new-old):",
                  numpy.unravel_index(delta.argmax(), npy_new_val.shape),
                  file=sio)
            print("  location of first 10 mismatches:",
                  numpy.transpose(numpy.nonzero(delta))[:10], file=sio)
            print("", file=sio)
        except Exception as e:
            print("(Numpy-hints failed with: %s)" % str(e), file=sio)
        print("  Hint: this can also be caused by a deficient "
              "values_eq_approx() or __eq__() implementation "
              "[which compared input values]", file=sio)
        return sio.getvalue()


class BadViewMap(DebugModeError):
    """
    Exception: Some perform() or c_code() created a memory alias
    that wasn't in the view_map.

    """

    def __init__(self, node, output_idx, out_storage,
                 in_alias_idx=None, out_alias_idx=None):
        super(BadViewMap, self).__init__()
        self.node = node
        self.output_idx = output_idx
        self.out_storage = out_storage
        self.in_alias_idx = in_alias_idx
        self.out_alias_idx = out_alias_idx

    def __str__(self):
        sio = StringIO()
        print("  node:", self.node, file=sio)
        print("  node.inputs:", [(str(i), id(i))
                                 for i in self.node.inputs], file=sio)
        print("  node.outputs:", [(str(i), id(i))
                                  for i in self.node.outputs], file=sio)
        print("  view_map:", getattr(self.node.op, 'view_map', {}), file=sio)
        print("  destroy_map:", getattr(self.node.op,
                                        'destroy_map', {}), file=sio)
        print("  aliased output:", self.output_idx, file=sio)
        print("  aliased output storage:", self.out_storage, file=sio)
        if self.in_alias_idx:
            print("  aliased to inputs:", self.in_alias_idx, file=sio)
        if self.out_alias_idx:
            print("  aliased to outputs:", self.out_alias_idx, file=sio)
        return sio.getvalue()


class StochasticOrder(DebugModeError):
    """
    Exception: Repeated Optimizations of the same graph do not give
    identical results.

    The most common cause is that an Optimization iterates over some
    objects in a memory-address-dependent order (such as id() or
    object.hash()).  If you see this error and you think it is related
    to optimizations within Theano, email theano-dev with the message
    attached to this exception.

    """
    pass


class InvalidValueError(DebugModeError):
    """
    Exception: some Op an output value that is inconsistent with
    the Type of that output.

    """

    def __init__(self, r, v, client_node=None, hint='none',
                 specific_hint='none'):
        super(InvalidValueError, self).__init__()
        self.r = r
        self.v = v
        self.client_node = client_node
        self.hint = hint
        self.specific_hint = specific_hint

    def __str__(self):
        r, v = self.r, self.v
        type_r = r.type
        type_v = type(v)
        v_val = str(v)[0:100]
        v_dtype = 'N/A'
        v_shape = 'N/A'
        v_min = 'N/A'
        v_max = 'N/A'
        v_isfinite = 'N/A'
        try:
            v_shape = v.shape
            v_dtype = v.dtype
            v_min = v.min()
            v_max = v.max()
            v_isfinite = numpy.all(numpy.isfinite(v))
        except Exception:
            pass
        client_node = self.client_node
        hint = self.hint
        specific_hint = self.specific_hint
        context = debugprint(r, prefix='  ', depth=12,
                             file=StringIO()).getvalue()
        return """InvalidValueError
        type(variable) = %(type_r)s
        variable       = %(r)s
        type(value)    = %(type_v)s
        dtype(value)   = %(v_dtype)s
        shape(value)   = %(v_shape)s
        value          = %(v_val)s
        min(value)     = %(v_min)s
        max(value)     = %(v_max)s
        isfinite       = %(v_isfinite)s
        client_node    = %(client_node)s
        hint           = %(hint)s
        specific_hint  = %(specific_hint)s
        context        = ...\n%(context)s
        """ % locals()



def char_from_number(number):
    """
    Converts number to string by rendering it in base 26 using
    capital letters as digits.

    """

    base = 26

    rval = ""

    if number == 0:
        rval = 'A'

    while number != 0:
        remainder = number % base
        new_char = chr(ord('A') + remainder)
        rval = new_char + rval
        number //= base

    return rval


def debugprint(r, prefix='', depth=-1, done=None, print_type=False,
               file=sys.stdout, print_destroy_map=False,
               print_view_map=False, order=None, ids='CHAR',
               stop_on_name=False, prefix_child=None,
               scan_ops=None, profile=None,
               scan_inner_to_outer_inputs=None, smap=None):
    """
    Print the graph leading to `r` to given depth.

    Parameters
    ----------
    r
        Variable instance.
    prefix
        Prefix to each line (typically some number of spaces).
    depth
        Maximum recursion depth (Default -1 for unlimited).
    done
        dict of Apply instances that have already been printed and their
        associated printed ids.
    print_type
        Whether to print the Variable type after the other infos.
    file
        File-like object to which to print.
    print_destroy_map
        Whether to print the op destroy_map after other info.
    print_view_map
        Whether to print the op view_map after other info.
    order
        If not empty will print the index in the toposort.
    ids
        How do we print the identifier of the variable :
        id - print the python id value,
        int - print integer character,
        CHAR - print capital character,
        "" - don't print an identifier.
    stop_on_name
        When True, if a node in the graph has a name, we don't print anything
        below it.
    scan_ops
        Scan ops in the graph will be added inside this list for later printing
        purposes.
    scan_inner_to_outer_inputs
        A dictionary mapping a scan ops inner function inputs to the scan op
        inputs (outer inputs) for printing purposes.
    smap
        None or the storage_map when printing an Theano function.
    """
    if depth == 0:
        return

    if order is None:
        order = []

    if done is None:
        done = dict()

    if scan_ops is None:
        scan_ops = []

    if print_type:
        type_str = ' <%s>' % r.type
    else:
        type_str = ''

    if prefix_child is None:
        prefix_child = prefix

    def get_id_str(obj):
        if obj in done:
            id_str = done[obj]
        elif ids == "id":
            id_str = "[id %s]" % str(id(r))
        elif ids == "int":
            id_str = "[id %s]" % str(len(done))
        elif ids == "CHAR":
            id_str = "[id %s]" % char_from_number(len(done))
        elif ids == "":
            id_str = ""
        done[obj] = id_str

        return id_str

    if hasattr(r.owner, 'op'):
        a = r.owner

        r_name = getattr(r, 'name', '')
        if r_name is None:
            r_name = ''

        if print_destroy_map:
            destroy_map_str = str(getattr(r.owner.op, 'destroy_map', ''))
        else:
            destroy_map_str = ''

        if print_view_map:
            view_map_str = str(getattr(r.owner.op, 'view_map', ''))
        else:
            view_map_str = ''
        if destroy_map_str and destroy_map_str != '{}':
            destroy_map_str = 'd=' + destroy_map_str
        if view_map_str and view_map_str != '{}':
            view_map_str = 'v=' + view_map_str

        o = ''
        if order:
            o = str(order.index(r.owner))

        already_printed = a in done  # get_id_str put it in the dict
        id_str = get_id_str(a)

        if len(a.outputs) == 1:
            idx = ""
        else:
            idx = ".%i" % a.outputs.index(r)
        data = ""
        if smap:
            data = " " + str(smap.get(a.outputs[0], ''))
        if profile is None or a not in profile.apply_time:
            print('%s%s%s %s%s \'%s\' %s %s %s%s' % (prefix, a.op,
                                                     idx,
                                                     id_str, type_str,
                                                     r_name,
                                                     destroy_map_str,
                                                     view_map_str,
                                                     o, data), file=file)
        else:
            op_time = profile.apply_time[a]
            op_time_percent = (op_time / profile.fct_call_time) * 100
            tot_time_dict = profile.compute_total_times()
            tot_time = tot_time_dict[a]
            tot_time_percent = (tot_time_dict[a] / profile.fct_call_time) * 100

            if len(a.outputs) == 1:
                idx = ""
            else:
                idx = ".%i" % a.outputs.index(r)
            print("%s%s%s %s%s '%s' %s %s %s%s --> "
                  "%8.2es %4.1f%% %8.2es %4.1f%%"
                  % (prefix, a.op,
                     idx,
                     id_str, type_str,
                     r_name,
                     destroy_map_str,
                     view_map_str,
                     o, data,
                     op_time,
                     op_time_percent,
                     tot_time,
                     tot_time_percent), file=file)

        if not already_printed:
            if (not stop_on_name or
                    not (hasattr(r, 'name') and r.name is not None)):
                new_prefix = prefix_child + ' |'
                new_prefix_child = prefix_child + ' |'

                for idx, i in enumerate(a.inputs):
                    if idx == len(a.inputs) - 1:
                        new_prefix_child = prefix_child + '  '

                    if hasattr(i, 'owner') and hasattr(i.owner, 'op'):
                        if isinstance(i.owner.op,
                                      theano.scan_module.scan_op.Scan):
                            scan_ops.append(i)

                    debugprint(
                        i, new_prefix, depth=depth - 1, done=done,
                        print_type=print_type, file=file, order=order,
                        ids=ids, stop_on_name=stop_on_name,
                        prefix_child=new_prefix_child, scan_ops=scan_ops,
                        profile=profile,
                        scan_inner_to_outer_inputs=scan_inner_to_outer_inputs,
                        smap=smap)
    else:
        if scan_inner_to_outer_inputs is not None and\
           r in scan_inner_to_outer_inputs:

            id_str = get_id_str(r)
            outer_r = scan_inner_to_outer_inputs[r]

            if hasattr(outer_r.owner, 'op'):
                outer_id_str = get_id_str(outer_r.owner)
            else:
                outer_id_str = get_id_str(outer_r)
            print('%s%s %s%s -> %s' % (prefix, r, id_str, type_str,
                                       outer_id_str), file=file)
        else:
            data = ""
            if smap:
                data = " " + str(smap.get(r, ''))
            id_str = get_id_str(r)
            print('%s%s %s%s%s' % (prefix, r, id_str,
                                   type_str, data),
                  file=file)

    return file


def _optcheck_fgraph(input_specs, output_specs, accept_inplace=False):
    """
    Create a FunctionGraph for debugging.

    Parameters
    ----------
    input_specs: WRITEME
        fgraph inputs.
    output_specs: WRITEME
        fgraph outputs.
    accept_inplace : bool
        Are inplace ops permitted in the original graph?

    Returns
    -------
    FunctionGraph
        A new FunctionGraph with a cloned graph, with debugging `Feature`
        instances already installed.

    """
    orig_inputs = [spec.variable for spec in input_specs]
    updates = [spec.update for spec in input_specs if spec.update]
    orig_outputs = [spec.variable for spec in output_specs] + updates

    equivalence_tracker = _VariableEquivalenceTracker()
    fgraph = gof.fg.FunctionGraph(orig_inputs, orig_outputs,
                                  features=[equivalence_tracker])

    if not accept_inplace:
        for node in fgraph.apply_nodes:
            if getattr(node.op, 'destroy_map', None):
                raise TypeError("Graph must not contain inplace operations",
                                node)
    else:
        for node in fgraph.apply_nodes:
            if getattr(node.op, 'destroy_map', None):
                fgraph.attach_feature(gof.DestroyHandler())
                break

    fgraph.attach_feature(Supervisor(
        input for spec, input in zip(input_specs, fgraph.inputs)
        if not (spec.mutable or (hasattr(fgraph, 'destroyers') and
                                 fgraph.destroyers(input)))))

    for feature in std_fgraph.features:
        fgraph.attach_feature(feature())

    return fgraph, list(map(SymbolicOutput, updates)), equivalence_tracker


class DataDestroyed():
    pass

data_destroyed = DataDestroyed()


def check_eq(var, val1, val2):
    if hasattr(var.tag, 'values_eq_approx'):
        return var.tag.values_eq_approx(val1, val2)
    else:
        return var.type.values_eq_approx(val1, val2)


def _check_inputs(node, storage_map, r_vals, dr_vals, active_nodes,
                  clobber_dr_vals=True,
                  perform=None, warn_input_not_reused=True):
    """
    Raise BadDestroyMap if necessary, update dr_vals.

    Returns a list of output variables that actually worked inplace
    (their value is aliased to the value of at least one input).

    It modify the storage_map to remove node.inputs variable that have
    been destroyed.

    """
    destroyed_idx_list = []
    destroy_map = getattr(node.op, 'destroy_map', {})
    for o_pos, i_pos_list in iteritems(destroy_map):
        destroyed_idx_list.extend(i_pos_list)
    destroyed_res_list = [node.inputs[i] for i in destroyed_idx_list]

    actually_inplace_outputs = []
    dmap = getattr(node.op, 'destroy_map', {})
    for oo, ii in iteritems(dmap):
        var = node.outputs[oo]
        out_var = storage_map[var][0]
        in_var = storage_map[node.inputs[ii[0]]][0]
        if (hasattr(var.type, 'may_share_memory') and
                var.type.may_share_memory(out_var, in_var)):
            actually_inplace_outputs.append(node.outputs[oo])

        if warn_input_not_reused and destroyed_res_list:
            if isinstance(node.op, OutputGuard):
                continue
            if out_var is not in_var:
                _logger.warning("Optimization Warning: input idx %d marked "
                                "as destroyed was not changed for node '%s'",
                                ii[0], str(node))

    vmap = getattr(node.op, 'view_map', {})
    for oo, ii in iteritems(vmap):
        var = node.outputs[oo]
        out_var = storage_map[var][0]
        in_var = storage_map[node.inputs[ii[0]]][0]
        may_share = (hasattr(var.type, 'may_share_memory') and
                     var.type.may_share_memory(out_var, in_var))
        if may_share:
            actually_inplace_outputs.append(node.outputs[oo])

        if warn_input_not_reused:
            if getattr(out_var, 'size', 2) <= 1:
                continue
            if isinstance(node.op, OutputGuard):
                continue
            if not may_share:
                _logger.warning("Optimization Warning: input idx %d marked "
                                "as viewed but new memory allocated by node "
                                "'%s'", ii[0], str(node))

    for r_idx, r in enumerate(node.inputs):
        if not r.type.values_eq(r_vals[r], storage_map[r][0]):
            if r in destroyed_res_list:
                if node in active_nodes:
                    if dr_vals.get(r, (0, node))[1] is not node:
                        raise Exception('failure in topological ordering')
                    if clobber_dr_vals:
                        dr_vals[r] = (storage_map[r][0], node)
                    storage_map[r][0] = data_destroyed
            else:
                raise BadDestroyMap(node, r_idx, r_vals[r],
                                    storage_map[r][0], perform)

    return actually_inplace_outputs


def _check_viewmap(node, storage_map):
    """
    This functions raises a BadViewMap exception when it detects the
    following:
    - Output node storages aliased to input storage, with no declaration
      in view_map.
    - If not aliased to an input, check if two outputs are aliased together
      and used subsequently in the graph.

    """

    for oi, onode in enumerate(node.outputs):

        good_alias, bad_alias = {}, {}
        outstorage = storage_map[onode][0]

        view_map = getattr(node.op, 'view_map', {})
        destroy_map = getattr(node.op, 'destroy_map', {})


        for ii, inode in enumerate(node.inputs):
            in_storage = storage_map[inode][0]
            if in_storage is data_destroyed:
                continue
            if (hasattr(inode.type, 'may_share_memory') and
                    inode.type.may_share_memory(outstorage, in_storage)):

                nodeid = id(inode)
                bad_alias[nodeid] = ii

                if ([ii] == view_map.get(oi, None) or
                        [ii] == destroy_map.get(oi, None)):

                    good_alias[nodeid] = bad_alias.pop(nodeid)

        for key, val in iteritems(good_alias):
            bad_alias.pop(key, None)
        if bad_alias:
            raise BadViewMap(node, oi, outstorage, list(bad_alias.values()))

        if not good_alias and _is_used_in_graph(onode):
            for other_oi, other_onode in enumerate(node.outputs):
                if other_oi == oi:
                    continue

                other_storage = storage_map[other_onode][0]
                if (_is_used_in_graph(other_onode) and
                    hasattr(other_onode.type, 'may_share_memory') and
                    other_onode.type.may_share_memory(outstorage,
                                                      other_storage)):
                    raise BadViewMap(node, oi, outstorage,
                                     out_alias_idx=other_oi)


def _is_used_in_graph(var):
    """

    Returns
    -------
    bool
        True if `var` is used by another node in the graph.

    """
    return not(var.clients == [('output', 1)] or var.clients == [])


def _check_strides_match(a, b, warn_err, op):
    """

    Parameters
    ----------
    warn_err
        If 0, no warning, if 1 warning, if 2 error.

    """
    if warn_err == 0:
        return

    try:
        strides_eq = a.strides == b.strides
    except Exception:
        return  # no strides

    if not strides_eq:
        e = TypeError('Stride mismatch', (a.shape, b.shape, a.strides,
                                          b.strides, str(op)))
        if warn_err == 2:
            raise e
        else:
            print('WARNING:', e, file=sys.stderr)


def _lessbroken_deepcopy(a):
    """

    Parameters
    ----------
    a
        Any object

    Returns
    -------
    object
        A copy of `a` that shares no internal storage with the original
        (a deep copy). This function handles numpy arrays specially, because
        copy.deepcopy() called on a 0-d array will return a numpy scalar,
        not an array.

    """
    from theano.gof.type import CDataType
    if type(a) in (numpy.ndarray, numpy.memmap):
        rval = a.copy()
    elif type(a) is CDataType._cdata_type:
        rval = a
    else:
        rval = copy.deepcopy(a)

    assert type(rval) == type(a), (type(rval), type(a))
    if isinstance(rval, numpy.ndarray):
        assert rval.dtype == a.dtype
    return rval


def _find_bad_optimizations0(order, reasons, r_vals):
    """
    Use a simple algorithm to find broken optimizations.

    This algorithm is simple to understand, but sometimes when there's
    a problem it identifies the wrong optimization as the culprit.
    The problem stems from the fact that results are not evaluated in
    chronological order (looking at when they were introduced to the
    graph).

    """
    for i, node in enumerate(order):
        for new_r in node.outputs:
            for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
                new_r_val = r_vals[new_r]
                r_val = r_vals[r]
                assert r.type == new_r.type

                if hasattr(new_r.tag, 'values_eq_approx'):
                    check = new_r.tag.values_eq_approx(r_val, new_r_val)
                elif hasattr(new_r, 'values_eq_approx'):
                    check = new_r.values_eq_approx(r_val, new_r_val)
                else:
                    check = r.type.values_eq_approx(r_val, new_r_val)
                if not check:
                    raise BadOptimization(old_r=r,
                                          new_r=new_r,
                                          old_r_val=r_val,
                                          new_r_val=new_r_val,
                                          reason=reason,
                                          old_graph=old_graph_str,
                                          new_graph=new_graph_str)


def _find_bad_optimizations1(order, reasons, r_vals):

    equivalence_sets = {}
    program_position = {}  # node -> order idx

    for i, node in enumerate(order):
        program_position[node] = i
        for new_r in node.outputs:
            equivalence_sets.setdefault(new_r, set([new_r]))
            for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
                equivalence_sets[new_r].update(equivalence_sets.setdefault(
                    r, set([r])))
                for er in equivalence_sets[r]:
                    equivalence_sets[er] = equivalence_sets[new_r]

    equivalence_sets_broken = {}  # id(set) -> Bool
    there_is_a_problem = False
    for r, r_equiv in iteritems(equivalence_sets):
        if id(r_equiv) not in equivalence_sets_broken:
            equivalence_sets_broken[id(r_equiv)] = False
            re0 = None
            for re in r_equiv:
                if re0:
                    new_r_val = r_vals[re]
                    r_val = r_vals[re0]
                    assert re.type == re0.type
                    if not re.type.values_eq_approx(r_val, new_r_val):
                        equivalence_sets_broken[id(r_equiv)] = True
                        there_is_a_problem = True
                re0 = re

    if there_is_a_problem:
        first_broken_set = None
        for i, node in enumerate(order):
            for r in node.outputs:
                r_equiv = equivalence_sets[r]
                if equivalence_sets_broken[id(r_equiv)]:
                    first_broken_set = r_equiv
        print(first_broken_set)
        raise Exception('broken')


def _find_bad_optimizations2(order, reasons, r_vals):
    """
    Use a simple algorithm to find broken optimizations.

    This algorithm is simple to understand, but sometimes when there's
    a problem it identifies the wrong optimization as the culprit.
    The problem stems from the fact that results are not evaluated in
    chronological order (looking at when they were introduced to the
    graph).

    """

    checked_variables = set()

    def check_variable_norec(new_r):
        """
        Verify that `r` has the same value as the results it replaces.

        """
        for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
            new_r_val = r_vals[new_r]
            r_val = r_vals[r]

            if (r.type != new_r.type) or (not r.type.values_eq_approx(
                    r_val, new_r_val)):
                raise BadOptimization(old_r=r,
                                      new_r=new_r,
                                      old_r_val=r_val,
                                      new_r_val=new_r_val,
                                      reason=reason,
                                      old_graph=old_graph_str,
                                      new_graph=new_graph_str)

    def check_variable(r):
        if r in checked_variables:
            return
        checked_variables.add(r)

        list_of_vars = [old_r for (reason, old_r, olds, news) in reasons[r]]
        if (None is not r.owner):
            list_of_vars += r.owner.inputs

        for var_that_could_make_r_look_bad in list_of_vars:
            check_variable(var_that_could_make_r_look_bad)

        check_variable_norec(r)

    for i, node in enumerate(order):
        for new_r in node.outputs:
            check_variable(new_r)

_find_bad_optimizations = _find_bad_optimizations0


def _get_preallocated_maps(node, thunk, prealloc_modes, def_val,
                           storage_map, r_vals, dr_vals, perform,
                           active_order_set, inplace_outs, init_outputs):
    """
    Preallocate outputs in different memory layouts.

    """

    from theano.tensor import TensorType
    from theano.sandbox.cuda import cuda_available, CudaNdarrayType
    if cuda_available:
        from theano.sandbox.cuda import CudaNdarray
        from theano.sandbox.cuda import dimshuffle as cuda_dimshuffle


    considered_outputs = []
    for r in node.outputs:
        if r not in inplace_outs:
            considered_outputs.append(r)

    if 'initial' in prealloc_modes or 'ALL' in prealloc_modes:
        initial_outputs = {}
        for r in considered_outputs:
            if r in init_outputs:
                initial_outputs[r] = init_outputs[r]

        if initial_outputs:
            yield ('initial', initial_outputs)

    if 'previous' in prealloc_modes or 'ALL' in prealloc_modes:
        reuse_outputs = {}
        for r in considered_outputs:
            new_r = _lessbroken_deepcopy(r_vals[r])
            reuse_outputs[r] = r_vals[r]
            r_vals[r] = new_r

        for r in considered_outputs:
            if isinstance(r.type, (TensorType, CudaNdarrayType)):
                reuse_outputs[r][...] = numpy.asarray(
                    def_val).astype(r.type.dtype)

        if reuse_outputs:
            yield ('previous', reuse_outputs)
        del reuse_outputs

    if 'c_contiguous' in prealloc_modes or 'ALL' in prealloc_modes:
        c_cont_outputs = {}
        for r in considered_outputs:
            if isinstance(r.type, (TensorType, CudaNdarrayType)):
                new_buf = r.type.value_zeros(r_vals[r].shape)
                new_buf[...] = numpy.asarray(def_val).astype(r.type.dtype)

                c_cont_outputs[r] = new_buf

        if len(c_cont_outputs):
            yield ('c_contiguous', c_cont_outputs)
            del c_cont_outputs

    if 'f_contiguous' in prealloc_modes or 'ALL' in prealloc_modes:
        f_cont_outputs = {}
        for r in considered_outputs:
            if isinstance(r.type, (TensorType, CudaNdarrayType)):
                new_buf = numpy.zeros(
                    shape=r_vals[r].shape,
                    dtype=r_vals[r].dtype,
                    order='F')
                new_buf[...] = def_val
                if isinstance(r.type, CudaNdarrayType):
                    new_buf = CudaNdarray(new_buf.T)
                    new_buf = cuda_dimshuffle(
                        new_buf, reversed(list(range(new_buf.ndim))))

                f_cont_outputs[r] = new_buf

        if len(f_cont_outputs):
            yield ('f_contiguous', f_cont_outputs)
            del f_cont_outputs

    if ('strided' in prealloc_modes or
            'wrong_size' in prealloc_modes or
            'ALL' in prealloc_modes):
        max_ndim = 0
        rev_out_broadcastable = []
        for r in considered_outputs:
            if isinstance(r.type, (TensorType, CudaNdarrayType)):
                if max_ndim < r.ndim:
                    rev_out_broadcastable += [True] * (r.ndim - max_ndim)
                    max_ndim = r.ndim
                assert len(rev_out_broadcastable) == max_ndim

                for i, b in enumerate(r.broadcastable[::-1]):
                    rev_out_broadcastable[i] = rev_out_broadcastable[i] and b
        out_broadcastable = rev_out_broadcastable[::-1]

    if 'strided' in prealloc_modes or 'ALL' in prealloc_modes:
        check_ndim = config.DebugMode.check_preallocated_output_ndim
        init_strided = {}
        for r in considered_outputs:
            if isinstance(r.type, (TensorType, CudaNdarrayType)):
                buf_shape = []
                for s, b in zip(r_vals[r].shape, r.broadcastable):
                    if b or ((r.ndim - len(buf_shape)) > check_ndim):
                        buf_shape.append(s)
                    else:
                        buf_shape.append(s * 2)
                new_buf = r.type.value_zeros(buf_shape)
                new_buf[...] = numpy.asarray(def_val).astype(r.type.dtype)
                init_strided[r] = new_buf

        step_signs_list = []
        for b in out_broadcastable[-check_ndim:]:
            if b:
                step_signs_list.append((1,))
            else:
                step_signs_list.append((-1, 1))

        if all(out_broadcastable[:-check_ndim]):
            step_signs_list = [(1,)] + step_signs_list
        else:
            step_signs_list = [(-1, 1)] + step_signs_list

        for step_signs in itertools_product(*step_signs_list):
            for step_size in (1, 2):
                strided = {}

                steps = [step_signs[0]] * len(out_broadcastable[:-check_ndim])
                steps += [s * step_size for s in step_signs[1:]]

                name = 'strided%s' % str(tuple(steps))
                for r in considered_outputs:
                    if r in init_strided:
                        strides = []
                        shapes = []
                        for i, size in enumerate(r_vals[r].shape):
                            shapes.append(slice(None, size, None))
                            strides.append(slice(None, None, steps[i]))

                        r_buf = init_strided[r]

                        if r_buf.ndim > 0:
                            r_buf = r_buf[tuple(strides)][tuple(shapes)]
                        assert r_buf.shape == r_vals[r].shape

                        r_buf[...] = numpy.asarray(def_val).astype(r_buf.dtype)
                        strided[r] = r_buf

                if strided:
                    yield (name, strided)
                del strided

    if 'wrong_size' in prealloc_modes or 'ALL' in prealloc_modes:
        for dim, b in enumerate(out_broadcastable):
            if b:
                continue

            shape_diff = [0] * max_ndim
            for diff in (-1, 1):
                shape_diff[dim] = diff

                wrong_size = {}
                name = 'wrong_size%s' % str(tuple(shape_diff))

                for r in considered_outputs:
                    if isinstance(r.type, (TensorType, CudaNdarrayType)):
                        r_shape_diff = shape_diff[:r.ndim]
                        out_shape = [max((s + sd), 0)
                                     for s, sd in zip(r_vals[r].shape,
                                                      r_shape_diff)]
                        new_buf = r.type.value_zeros(out_shape)
                        new_buf[...] = numpy.asarray(
                            def_val).astype(r.type.dtype)
                        wrong_size[r] = new_buf

                if wrong_size:
                    yield (name, wrong_size)
                del wrong_size


def _check_preallocated_output(node, thunk, prealloc_modes, def_val,
                               storage_map, r_vals, dr_vals, perform,
                               active_order_set, inplace_outs, init_outputs):
    """
    Try to apply thunk() on different output storages.

    """

    try:
        changed_inner_mode = False
        if type(getattr(node, 'op', None)) in ops_with_inner_function:
            fn_attr_name = ops_with_inner_function[type(node.op)]
            fn = getattr(node.op, fn_attr_name, None)
            if (not fn or
                    not hasattr(fn, 'maker') or
                    not hasattr(fn.maker, 'mode')):
                _logger.warn('Expected theano function not found in %s.%s',
                             node.op, fn_attr_name)
            else:
                if isinstance(fn.maker.mode, DebugMode):
                    backup_mode = fn.maker.mode
                    new_mode = copy.copy(backup_mode)
                    new_mode.check_py_code = False
                    new_mode.check_isfinite = False
                    new_mode.require_matching_strides = 0
                    new_mode.check_preallocated_output = []
                    new_mode.stability_patience = 1
                    fn.maker.mode = new_mode
                    changed_inner_mode = True
                    _logger.info('changing inner mode')

        aliased_inputs = set()
        dmap = getattr(node.op, 'destroy_map', {})
        vmap = getattr(node.op, 'view_map', {})
        for i, r in enumerate(node.inputs):
            if any(i in v for v in chain(itervalues(dmap), itervalues(vmap))):
                aliased_inputs.add(r)

        _logger.debug('starting preallocated output checking')
        for (name, out_map) in _get_preallocated_maps(
                node, thunk, prealloc_modes, def_val, storage_map, r_vals,
                dr_vals, perform, active_order_set, inplace_outs,
                init_outputs):
            _logger.debug('  name = %s', name)

            thunk_name = '%s with %s output' % (perform, name)

            if not out_map:
                _logger.warn('%s: out_map is empty', name)
                continue

            for r in aliased_inputs:
                storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])

            for r in node.outputs:
                storage_map[r][0] = out_map.get(r, None)

            thunk()

            for r in node.outputs:
                if not r.type.is_valid_value(storage_map[r][0]):
                    raise InvalidValueError(
                        r, storage_map[r][0],
                        hint=thunk_name,
                        specific_hint=r.type.value_validity_msg(
                            storage_map[r][0]))

            _check_inputs(node, storage_map, r_vals, dr_vals, active_order_set,
                          clobber_dr_vals=False,
                          perform=thunk_name,
                          warn_input_not_reused=False)

            _check_viewmap(node, storage_map)

            for r in node.outputs:
                if not check_eq(r, r_vals[r], storage_map[r][0]):
                    inputs_val = [storage_map[inp][0] for inp in
                                  r.owner.inputs]
                    raise BadThunkOutput(r,
                                         thunk1='Reference value',
                                         val1=r_vals[r],
                                         thunk2=thunk_name,
                                         val2=storage_map[r][0],
                                         inputs_val=inputs_val)

            for r in node.outputs:
                storage_map[r][0] = None

        _logger.debug('finished preallocated output checking')
    finally:
        if changed_inner_mode:
            _logger.info('changing mode back')
            fn.maker.mode = backup_mode


class _FunctionGraphEvent(object):
    """
    A record of an event in the life of an FunctionGraph.

    The __eq__ function is important here, as it is the basis for
    comparing optimization runs.

    """

    kind = ""
    """
    One of 'import', 'change', 'prune'.

    """

    node = None
    """
    Either 'output' or an Apply instance.

    """

    op = None
    """Either 'output' or an Op instance"""

    idx = None
    """
    Change events involve an position index of the input variable.

    """

    reason = None
    """
    Change events sometimes have a reason.

    """

    def __init__(self, kind, node, idx=None, reason=None):
        self.kind = kind
        if node == 'output':
            self.node = 'output'
            self.op = 'output'
        else:
            self.node = node
            self.op = node.op
        self.idx = idx
        self.reason = reason

    def __str__(self):
        if self.kind == 'change':
            if (self.op != 'output'):
                msg = str(len(self.node.inputs))
            else:
                msg = ''

            return ' '.join(['change',
                             self.reason,
                             str(self.op),
                             str(self.idx),
                             msg])
        else:
            return str(self.__dict__)

    def __eq__(self, other):
        rval = type(self) == type(other)
        if rval:
            for attr in ['kind', 'op', 'idx', 'reason']:
                rval = rval and getattr(self, attr) == getattr(other, attr)
        return rval

    def __ne__(self, other):
        return not (self == other)


class _VariableEquivalenceTracker(object):
    """
    A FunctionGraph Feature that keeps tabs on an FunctionGraph and
    tries to detect problems.

    """

    fgraph = None
    """WRITEME"""

    equiv = None
    """WRITEME"""

    active_nodes = None
    """WRITEME"""

    inactive_nodes = None
    """WRITEME"""

    all_variables_ever = None
    """WRITEME"""

    reasons = None
    """WRITEME"""

    replaced_by = None
    """WRITEME"""

    event_list = None
    """WRITEME"""

    def __init__(self):
        self.fgraph = None

    def on_attach(self, fgraph):
        assert self.fgraph is None
        self.equiv = {}
        self.active_nodes = set()
        self.inactive_nodes = set()
        self.fgraph = fgraph
        self.all_variables_ever = []
        self.reasons = {}
        self.replaced_by = {}
        self.event_list = []
        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def on_detach(self, fgraph):
        assert fgraph is self.fgraph
        self.fgraph = None

    def on_prune(self, fgraph, node, reason):
        self.event_list.append(_FunctionGraphEvent('prune', node,
                                                   reason=reason))
        assert node in self.active_nodes
        assert node not in self.inactive_nodes
        self.active_nodes.remove(node)
        self.inactive_nodes.add(node)

    def on_import(self, fgraph, node, reason):
        self.event_list.append(_FunctionGraphEvent('import', node,
                                                   reason=reason))

        assert node not in self.active_nodes
        self.active_nodes.add(node)

        if node in self.inactive_nodes:
            self.inactive_nodes.remove(node)
            for r in node.outputs:
                assert r in self.equiv
        else:
            for r in node.outputs:
                assert r not in self.equiv
                self.equiv[r] = set([r])
                self.all_variables_ever.append(r)
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])
            for r in node.inputs:
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        self.event_list.append(_FunctionGraphEvent('change', node,
                                                   reason=str(reason), idx=i))

        self.reasons.setdefault(new_r, [])
        self.replaced_by.setdefault(new_r, [])

        append_reason = True
        for tup in self.reasons[new_r]:
            if tup[0] == reason and tup[1] is r:
                append_reason = False

        if append_reason:
            done = dict()
            self.reasons[new_r].append(
                (reason,
                 r,
                 debugprint(r, prefix='  ', depth=6,
                            file=StringIO(), done=done).getvalue(),
                 debugprint(new_r, prefix='  ', depth=6,
                            file=StringIO(), done=done).getvalue()))
            self.replaced_by[r].append((reason, new_r))

        if r in self.equiv:
            r_set = self.equiv[r]
        else:
            r_set = self.equiv.setdefault(r, set([r]))
            self.all_variables_ever.append(r)

        if new_r in self.equiv:
            new_r_set = self.equiv[new_r]
        else:
            new_r_set = self.equiv.setdefault(new_r, set([new_r]))
            self.all_variables_ever.append(new_r)

        assert new_r in new_r_set
        assert r in r_set

        r_set.update(new_r_set)
        for like_new_r in new_r_set:
            self.equiv[like_new_r] = r_set
            assert like_new_r in r_set

        assert self.equiv[r] is r_set
        assert self.equiv[new_r] is r_set

    def printstuff(self):
        for key in self.equiv:
            print(key)
            for e in self.equiv[key]:
                print('  ', e)


default_make_thunk = [get_unbound_function(theano.gof.Op.make_thunk)]


class _DummyLinker(object):
    def clone(self, allow_gc=None):
        return self


class _Linker(gof.link.LocalLinker):
    """
    Special debugging linker.

    """

    def __init__(self, maker, schedule=None):
        super(gof.LocalLinker, self).__init__()
        self.fgraph = None
        self.maker = maker
        if schedule:
            self.schedule = schedule

    def accept(self, fgraph, no_recycling=None):
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            assert type(self) is _Linker
            return type(self)(maker=self.maker).accept(fgraph, no_recycling)
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        return self

    def make_all(self, profiler=None, input_storage=None,
                 output_storage=None, storage_map=None):
        from theano.tensor import TensorType  # to set filter_check_isfinite

        fgraph = self.fgraph
        input_storage_ = input_storage
        output_storage_ = output_storage

        fgraph_equiv = fgraph.equivalence_tracker
        order_outputs = copy.copy(fgraph_equiv.all_variables_ever)
        del fgraph_equiv
        order_outputs.reverse()
        order = graph.io_toposort(fgraph.inputs, order_outputs)

        active_order = self.schedule(fgraph)
        active_order_set = set(active_order)

        no_recycling = []

        input_storage, output_storage, storage_map = link.map_storage(
            fgraph, order, input_storage_, output_storage_, storage_map)

        thunks_py = []  # python thunks
        thunks_c = []  # c thunks

        for node in order:
            compute_map = {}
            for k in node.inputs:
                compute_map[k] = [True]
            for k in node.outputs:
                compute_map[k] = [False]

            thunk_other = None

            if (get_unbound_function(node.op.make_thunk) not in
                    default_make_thunk):
                thunk = node.op.make_thunk(node,
                                           storage_map,
                                           compute_map,
                                           no_recycling)
                thunk.inputs = [storage_map[v] for v in node.inputs]
                thunk.outputs = [storage_map[v] for v in node.outputs]
                thunk_other = thunk
            else:
                new_node = node.op.prepare_node(node, storage_map, compute_map)
                if new_node is not None:
                    node = new_node

            debug = hasattr(node.op, 'debug_perform')

            try:
                if not self.maker.mode.check_c_code or debug:
                    raise utils.MethodNotDefined()
                if not isinstance(node.op, gof.op.Op):
                    raise utils.MethodNotDefined()

                thunk = node.op.make_c_thunk(node, storage_map, compute_map,
                                             no_recycling)
                thunks_c.append(thunk)
            except (NotImplementedError, utils.MethodNotDefined):
                thunks_c.append(None)

            if (((self.maker.mode.check_py_code or thunks_c[-1] is None) and
                 node.op.perform.__code__ != gof.op.PureOp.perform.__code__) or
                    debug):
                thunk = node.op.make_py_thunk(node, storage_map, compute_map,
                                              no_recycling, debug=debug)
                thunks_py.append(thunk)
            else:
                thunks_py.append(None)

            if not self.maker.mode.check_c_code and thunks_py[-1] is None:
                _logger.warn("Op %s doesn't have a perform, "
                             "forcing check of the C code" % node.op)
                thunk = node.op.make_c_thunk(node, storage_map, compute_map,
                                             no_recycling)
                thunks_c[-1] = thunk

            if thunk_other is not None:
                if thunks_py[-1] is None:
                    thunks_py[-1] = thunk_other
                elif thunks_c[-1] is None:
                    thunks_c[-1] = thunk_other
                else:
                    _logger.warn("We won't check the perform function "
                                 "of node '%s' but we will check its "
                                 "make_thunk function" % node)
                    thunks_py[-1] = thunk_other

        if self.no_recycling is True:
            no_recycling_map = list(storage_map.values())
            no_recycling_map = utils.difference(no_recycling_map,
                                                input_storage)
        else:
            no_recycling_map = [storage_map[r] for r in self.no_recycling
                                if r not in fgraph.inputs]

        def_val = int(config.unittests.rseed)

        def f():
            _logger.debug("starting a DebugMode call")
            _logger.debug("self.maker.mode.check_preallocated_output: %s",
                          self.maker.mode.check_preallocated_output)
            for x in no_recycling_map:
                x[0] = None

            original_storage_map_keys = [r for r in storage_map
                                         if r.owner is None]

            try:
                r_vals = {}

                dr_vals = {}
                assert len(thunks_py) == len(order)

                _logger.debug("DEBUGMODE: transfer initial values")
                r_vals_initialized = []
                for r in storage_map:
                    if (r.owner is None):
                        if not r.type.is_valid_value(storage_map[r][0]):
                            if (storage_map[r][0] is None):
                                raise InvalidValueError(
                                    r, storage_map[r][0],
                                    hint=("Graph Input '%s' is missing" %
                                          str(r)))
                            raise InvalidValueError(
                                r, storage_map[r][0],
                                hint=("Graph Input '%s' has invalid value "
                                      "%s" % (r, storage_map[r][0])))
                        r_vals[r] = storage_map[r][0]
                        storage_map[r][0] = None
                        r_vals_initialized.append(r)

                init_outputs = {}
                for r in storage_map:
                    if r in fgraph.outputs:
                        if storage_map[r][0] is not None:
                            init_outputs[r] = storage_map[r][0]
                            storage_map[r][0] = None

                for r, s in iteritems(storage_map):
                    if s[0] is not None:
                        print(r, s)
                    assert s[0] is None

                for i, (thunk_py, thunk_c, node) in enumerate(zip(thunks_py,
                                                                  thunks_c,
                                                                  order)):
                    _logger.debug("%i - starting node %i %s", i, i, node)

                    for r in node.inputs:
                        assert isinstance(r, gof.Variable)
                        assert r in r_vals
                        storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])
                        if not r.type.is_valid_value(storage_map[r][0]):
                            raise InvalidValueError(r, storage_map[r][0],
                                                    client_node=node)

                    if thunk_py:
                        _logger.debug("%i - running thunk_py with None as "
                                      "output storage", i)
                        try:
                            thunk_py()
                        except (utils.MethodNotDefined, NotImplementedError):
                            thunk_py = None
                            thunks_py[i] = None
                        except Exception as e:
                            reason = fgraph.equivalence_tracker.reasons[
                                node.outputs[0]]
                            if not reason:
                                raise
                            opt = str(reason[0][0])
                            msg = (
                                "An optimization (probably %s) inserted an "
                                "apply node that raise an error." % opt +
                                "\nThe information we have about this "
                                "optimizations is:" + str(reason[0][1]) +
                                "\n" + reason[0][2] +
                                "\n\nThe original exception: \n" + str(e))
                            new_e = e.__class__(msg)
                            exc_type, exc_value, exc_trace = sys.exc_info()
                            exc_value = new_e
                            raise_with_op(node, thunk_c,
                                          (exc_type, exc_value, exc_trace))

                    if thunk_py:
                        for r in node.outputs:
                            if not r.type.is_valid_value(storage_map[r][0]):
                                hint2 = r.type.value_validity_msg(
                                    storage_map[r][0])
                                raise InvalidValueError(r, storage_map[r][0],
                                                        hint='perform output',
                                                        specific_hint=hint2)
                        warn_inp = config.DebugMode.warn_input_not_reused
                        py_inplace_outs = _check_inputs(
                            node, storage_map, r_vals, dr_vals,
                            active_order_set,
                            clobber_dr_vals=True, perform='py',
                            warn_input_not_reused=warn_inp)
                        _check_viewmap(node, storage_map)

                        for r in node.outputs:
                            assert r not in r_vals
                            r_vals[r] = storage_map[r][0]
                            storage_map[r][0] = None

                        if self.maker.mode.check_preallocated_output:
                            prealloc_modes = \
                                self.maker.mode.check_preallocated_output
                            _logger.debug(
                                '%i - calling _check_preallocated_output '
                                'with thunk_py', i)
                            _check_preallocated_output(
                                node=node,
                                thunk=thunk_py,
                                prealloc_modes=prealloc_modes,
                                def_val=def_val,
                                storage_map=storage_map,
                                r_vals=r_vals,
                                dr_vals=dr_vals,
                                perform='py',
                                active_order_set=active_order_set,
                                inplace_outs=py_inplace_outs,
                                init_outputs=init_outputs)

                        sys.stdout.flush()

                    if thunk_c:

                        clobber = True
                        if thunk_py:
                            dmap = getattr(node.op, 'destroy_map', {})
                            vmap = getattr(node.op, 'view_map', {})
                            for i, r in enumerate(node.inputs):
                                if any(i in v for v in chain(dmap.values(),
                                                             vmap.values())):
                                    storage_map[r][0] = _lessbroken_deepcopy(
                                        r_vals[r])

                            clobber = False

                        _logger.debug("%i - running thunk_c", i)
                        try:
                            thunk_c()
                        except Exception as e:
                            reason = fgraph.equivalence_tracker.reasons[
                                node.outputs[0]]
                            if not reason:
                                raise
                            opt = str(reason[0][0])
                            msg = (
                                "An optimization (probably %s) inserted "
                                "an apply node that raise an error." % opt +
                                "\nThe information we have about this "
                                "optimizations is:" + str(reason[0][1]) +
                                "\n" + reason[0][2] +
                                "\n\nThe original exception: \n" + str(e))
                            new_e = e.__class__(msg)
                            exc_type, exc_value, exc_trace = sys.exc_info()
                            exc_value = new_e
                            raise_with_op(node, thunk_c,
                                          (exc_type, exc_value, exc_trace))

                        for r in node.outputs:
                            if not r.type.is_valid_value(storage_map[r][0]):
                                raise InvalidValueError(r, storage_map[r][0],
                                                        hint='c output')

                            if thunk_py:
                                assert r in r_vals
                                _check_strides_match(
                                    r_vals[r], storage_map[r][0],
                                    self.maker.mode.require_matching_strides,
                                    node.op)

                        warn_inp = config.DebugMode.warn_input_not_reused
                        c_inplace_outs = _check_inputs(
                            node, storage_map, r_vals,
                            dr_vals, active_order_set,
                            clobber_dr_vals=clobber, perform='c',
                            warn_input_not_reused=warn_inp)

                        _check_viewmap(node, storage_map)

                        for r in node.outputs:
                            if r in r_vals:
                                if not check_eq(r, r_vals[r],
                                                storage_map[r][0]):
                                    inputs_val = [storage_map[inp][0]
                                                  for inp in r.owner.inputs]
                                    raise BadThunkOutput(
                                        r, thunk1='perform', val1=r_vals[r],
                                        thunk2='c_code',
                                        val2=storage_map[r][0],
                                        inputs_val=inputs_val)
                            else:
                                r_vals[r] = storage_map[r][0]
                            storage_map[r][0] = None

                        if self.maker.mode.check_preallocated_output:
                            prealloc_modes = \
                                self.maker.mode.check_preallocated_output

                            def thunk():
                                try:
                                    thunk_c()
                                except Exception:
                                    raise_with_op(node, thunk_c)
                            _logger.debug(
                                '%i - calling _check_preallocated_output '
                                'with thunk_c', i)
                            _check_preallocated_output(
                                node=node,
                                thunk=thunk,
                                prealloc_modes=prealloc_modes,
                                def_val=def_val,
                                storage_map=storage_map,
                                r_vals=r_vals,
                                dr_vals=dr_vals,
                                perform='c code',
                                active_order_set=active_order_set,
                                inplace_outs=c_inplace_outs,
                                init_outputs=init_outputs)

                        sys.stdout.flush()

                    for r in node.inputs:
                        storage_map[r][0] = None
                    _logger.debug("%i - done with node", i)
                    for r in node.outputs:
                        if r not in r_vals:
                            idx = order.index(node)
                            assert thunks_py[idx] is None, node
                            assert thunks_c[idx] is None, node
                            raise Exception("No code run for %s" % node)

                if False:
                    gc.collect()

                _find_bad_optimizations(order,
                                        fgraph.equivalence_tracker.reasons,
                                        r_vals)


                for r, s in iteritems(storage_map):
                    assert type(s) is list
                    assert s[0] is None

                for output, storage in zip(fgraph.outputs, output_storage):
                    storage[0] = r_vals[output]

                for r in r_vals:
                    if r.owner is None:
                        if r in fgraph.inputs:
                            assert (storage_map[r] is
                                    input_storage[fgraph.inputs.index(r)])
                        storage_map[r][0] = r_vals[r]

                for r in dr_vals:
                    assert dr_vals[r][0] is not None
                    if r.owner is None:
                        assert r in fgraph.inputs
                        if ((type(dr_vals[r][0]) in
                             (numpy.ndarray, numpy.memmap)) and
                            (dr_vals[r][0].dtype ==
                             storage_map[r][0].dtype) and
                            (dr_vals[r][0].shape ==
                             storage_map[r][0].shape)):
                            if len(dr_vals[r][0].shape):
                                storage_map[r][0][:] = dr_vals[r][0]
                            else:
                                storage_map[r][0].itemset(dr_vals[r][0])
                        else:
                            storage_map[r][0] = dr_vals[r][0]
            except Exception:
                for r in storage_map:
                    if r in original_storage_map_keys:
                        if r in r_vals_initialized:
                            storage_map[r][0] = r_vals[r]
                    else:
                        storage_map[r][0] = None
                raise

            for r in storage_map:
                if (r.owner is None):
                    if not r.type.is_valid_value(None):
                        assert storage_map[r][0] is not None


        def run_with_tensortype_filter_check(f):
            def deco():
                old_filter_checks_isfinite = TensorType.filter_checks_isfinite
                TensorType.filter_checks_isfinite = \
                    self.maker.mode.check_isfinite
                try:
                    return f()
                finally:
                    TensorType.filter_checks_isfinite = \
                        old_filter_checks_isfinite
            return deco

        f = run_with_tensortype_filter_check(f)
        f.storage_map = storage_map
        f.allow_gc = True
        assert len(fgraph.inputs) == len(input_storage)
        assert len(fgraph.outputs) == len(output_storage)
        return (f,
                [link.Container(input, storage, readonly=False)
                 for input, storage in zip(fgraph.inputs, input_storage)],
                [link.Container(output, storage, readonly=True)
                 for output, storage in zip(fgraph.outputs, output_storage)],
                thunks_py, order)


_NODEFAULT = ['NODEFAULT']


class _Maker(FunctionMaker):  # inheritance buys a few helper functions
    """
    Special debugging FunctionMaker.

    Parameters
    ----------
    inputs : list of SymbolicInput instances
    outputs : list of SymbolicOutput instances
        Outputs may also be a single Variable (not a list), in which case
        the functions produced by FunctionMaker will return their output
        value directly.
    accept_inplace
        True iff it is acceptable to have inplace operations in the graph from
        the inputs to the outputs.
    on_unused_input
        What to do if a variable in the 'inputs' list is not used in the
        graph. Possible values are 'raise', 'warn' and 'ignore'.
    output_keys
        If the outputs argument for theano.function was a list, then
        output_keys is None. If the outputs argument was a dict, then
        output_keys is a sorted list of the keys from that dict.

    Notes
    -----
    The constructor sets TensorType.filter_checks_isfinite when
    `mode.check_isfinite` is True.

    """

    verbose = 0
    """
    Verbosity level of compile-time and run-time checks. (Default 0: silent).

    """

    def __init__(self, inputs, outputs, mode,
                 accept_inplace=False,
                 function_builder=Function,
                 profile=None,
                 on_unused_input=None,
                 fgraph=None,  # If present the optimized graph. we ignore it.
                 output_keys=None):
        self.profile = profile
        optimizer = mode.optimizer
        unpack_single = False
        return_none = False
        if outputs is None:
            return_none = True
            outputs = []
        if not isinstance(outputs, (list, tuple)):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [self.wrap_in(i) for i in inputs]
        outputs = [self.wrap_out(o) for o in outputs]

        _inputs = gof.graph.inputs([o.variable for o in outputs] +
                                   [i.update for i in inputs
                                    if getattr(i, 'update', False)])

        self._check_unused_inputs(inputs, outputs, on_unused_input)

        indices = [[input] + self.expand_in(input, _inputs)
                   for input in inputs]

        for i in xrange(mode.stability_patience):
            fgraph, additional_outputs, equivalence_tracker = _optcheck_fgraph(
                inputs, outputs, accept_inplace)
            fgraph.equivalence_tracker = equivalence_tracker

            compute_test_value_orig = theano.config.compute_test_value
            try:
                theano.config.compute_test_value = \
                    theano.config.compute_test_value_opt
                optimizer(fgraph)

                theano.compile.function_module.insert_deepcopy(
                    fgraph, inputs, list(chain(outputs, additional_outputs)))
            finally:
                theano.config.compute_test_value = compute_test_value_orig

            if i == 0:
                fgraph0 = fgraph
            else:
                li = fgraph.equivalence_tracker.event_list
                l0 = fgraph0.equivalence_tracker.event_list
                if li != l0:
                    infolog = StringIO()
                    print("WARNING: Optimization process is unstable...",
                          file=infolog)
                    print("  (HINT: Ops that the nodes point to must compare "
                          "equal)", file=infolog)
                    print("(event index)  (one event trace)  (other event "
                          "trace)", file=infolog)
                    print("-------------------------------------------------"
                          "----", file=infolog)
                    for j in xrange(max(len(li), len(l0))):
                        if j >= len(li):
                            print('trailing event in optimization 0 :', j,
                                  file=infolog)
                            print('   ', str(l0[j]), file=infolog)
                        elif j >= len(l0):
                            print('trailing event in optimization', i, ':',
                                  j, file=infolog)
                            print('   ', str(li[j]), file=infolog)
                        elif li[j] != l0[j]:
                            print('non-equal optimization events', i, ':',
                                  j, file=infolog)
                            print('   ', str(l0[j]), file=infolog)
                            print('   ', str(li[j]), file=infolog)
                        else:
                            pass
                    raise StochasticOrder(infolog.getvalue())
                else:
                    if self.verbose:
                        print("OPTCHECK: optimization", i,
                              "of", len(li), "events was stable.",
                              file=sys.stderr)
        self.fgraph = fgraph

        linker = _Linker(self)


        no_borrow = [
            output
            for output, spec in izip(fgraph.outputs,
                                     outputs + additional_outputs)
            if not spec.borrow]
        if no_borrow:
            self.linker = linker.accept(
                fgraph,
                no_recycling=infer_reuse_pattern(fgraph, no_borrow))
        else:
            self.linker = linker.accept(fgraph)

        self.indices = indices
        self.inputs = inputs
        self.expanded_inputs = inputs
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.return_none = return_none
        self.accept_inplace = accept_inplace
        self.function_builder = function_builder
        self.mode = mode
        self.on_unused_input = on_unused_input  # Used for the pickling/copy
        self.output_keys = output_keys

    def create(self, defaults=None, trustme=False, storage_map=None):
        """
        Create a function.

        Parameters
        ----------
        defaults
            A list matching the inputs list and providing default values if the
            default for an input is None, then that input is a required input.
            For an input with an update, the default acts as initialization.
        trustme
            Disables some exceptions, used internally.

        """
        if defaults is None:
            defaults = [None] * len(self.inputs)
        input_storage = []
        _defaults = []

        for (input, indices, subinputs), default in izip(self.indices,
                                                         defaults):
            __default = default

            if isinstance(default, gof.Container):
                if indices is not None:
                    raise TypeError("Cannot take a Container instance as "
                                    "default for a SymbolicInputKit.")
                input_storage.append(default.storage)
                default = None
            elif isinstance(input, SymbolicInputKit):
                if isinstance(default, (list, tuple)) \
                        and all(isinstance(x, gof.Container) for x in default):
                    if len(default) == len(indices):
                        input_storage += [x.storage for x in default]
                    elif len(default) > len(indices):
                        input_storage += [default[i].storage for i in indices]
                    else:
                        raise ValueError(
                            'Not enough storage for SymbolicInputKit',
                            input, indices, default)
                    default = _NODEFAULT
                else:
                    input_storage += [[None] for i in indices]
            else:
                input_storage.append([None])


            if isinstance(input, SymbolicInputKit):
                if default is _NODEFAULT:
                    _defaults.append((False, False, None))
                elif default is None:
                    _defaults.append((True, True, None))
                else:
                    _defaults.append((False, False, default))
            elif input.update is not None:
                if default is None:
                    if trustme or isinstance(__default, gof.Container):
                        _defaults.append((False, False, None))
                    else:
                        raise ValueError(
                            "A default (initial) value is required for an "
                            "input which can update itself.", input)
                else:
                    _defaults.append((False, False, default))
            else:
                if default is None:
                    if trustme or isinstance(__default, gof.Container):
                        _defaults.append((False, False, None))
                    else:
                        _defaults.append((True, False, None))
                else:
                    _defaults.append((False, True, default))
        defaults = _defaults

        _fn, _i, _o = self.linker.make_thunk(input_storage=input_storage,
                                             storage_map=storage_map)
        fn = self.function_builder(_fn, _i, _o, self.indices,
                                   self.outputs, defaults, self.unpack_single,
                                   self.return_none, self.output_keys, self)
        return fn


def _pickle_DebugMode_Maker(maker):
    raise NotImplementedError('DebugMode is not picklable (yet)')
copyreg.pickle(_Maker, _pickle_DebugMode_Maker)



class DebugMode(Mode):
    """
    Evaluation Mode that detects internal theano errors.

    This mode catches several kinds of internal error:

    - Inconsistent outputs when calling the same Op twice with the same
      inputs, for instance if c_code and perform implementations, are
      inconsistent, or in case of incorrect handling of output memory
      (see `BadThunkOutput`).

    - A variable replacing another when their runtime values don't
      match.  This is a symptom of an incorrect optimization step, or
      faulty Op implementation (raises `BadOptimization`).

    - Stochastic optimization ordering (raises `StochasticOrder`).

    - Incomplete `destroy_map` specification (raises `BadDestroyMap`).

    - An op that returns an illegal value not matching the output
      Variable Type (raises InvalidValueError).

    Each of these exceptions inherits from the more generic `DebugModeError`.

    If there are no internal errors, this mode behaves like FAST_RUN
    or FAST_COMPILE, but takes a little longer and uses more memory.

    Raises
    ------
    DebugModeError
        If there are internal errors.

    Notes
    -----
    The work of debugging is implemented by the `_Maker`, `_Linker`,
    and `_VariableEquivalenceTracker` classes.

    """

    stability_patience = config.DebugMode.patience
    """
    When checking for the stability of optimization, recompile the
    graph this many times.

    """

    check_c_code = config.DebugMode.check_c
    """
    Should we evaluate (and check) the `c_code` implementations?

    """

    check_py_code = config.DebugMode.check_py
    """
    Should we evaluate (and check) the `perform` implementations?
    Always checked if no `c_code`.

    """

    check_isfinite = config.DebugMode.check_finite
    """
    Should we check for (and complain about) NaN/Inf ndarray elements?

    """

    require_matching_strides = config.DebugMode.check_strides
    """
    Should we check for (and complain about) Ops whose python and C
    outputs are ndarrays with different strides? (This can catch bugs,
    but is generally overly strict.) 0 no check, 1 warn, 2 err.

    """

    check_preallocated_output = config.DebugMode.check_preallocated_output
    check_preallocated_output = check_preallocated_output.split(':')
    """
    List of strings representing ways to pre-allocate output memory in
    tests.  Valid values are: "previous" (previously-returned memory),
    "c_contiguous", "f_contiguous", "strided" (positive and negative
    strides), "wrong_size" (larger and smaller dimensions), and "ALL"
    (all of the above).

    """

    def function_maker(self, i, o, m, *args, **kwargs):
        """
        Return an instance of `_Maker` which handles much of the debugging work.

        """
        assert m is self
        return _Maker(i, o, self, *args, **kwargs)

    def __init__(self,
                 optimizer='fast_run',
                 stability_patience=None,
                 check_c_code=None,
                 check_py_code=None,
                 check_isfinite=None,
                 check_preallocated_output=None,
                 require_matching_strides=None,
                 linker=_DummyLinker()):
        """
        If any of these arguments (except optimizer) is not None, it overrides
        the class default. The linker argument is not used. It is set there to
        allow Mode.requiring() and some other fct to work with DebugMode too.

        """

        if not isinstance(linker, _DummyLinker):
            raise Exception("DebugMode can only use its own linker! You "
                            "should not provide one.", linker)

        super(DebugMode, self).__init__(optimizer=optimizer,
                                        linker=linker)

        if stability_patience is not None:
            self.stability_patience = stability_patience

        if check_c_code is not None:
            self.check_c_code = check_c_code

        if check_py_code is not None:
            self.check_py_code = check_py_code

        if check_isfinite is not None:
            self.check_isfinite = check_isfinite

        if check_preallocated_output is not None:
            self.check_preallocated_output = check_preallocated_output[:]

        if require_matching_strides is not None:
            self.require_matching_strides = require_matching_strides

        if not (self.check_c_code or self.check_py_code):
            raise ValueError('DebugMode has to check at least one of c and py '
                             'code')

    def __str__(self):
        return "DebugMode(linker=%s, optimizer=%s)" % (
            self.provided_linker, self.provided_optimizer)


register_mode('DEBUG_MODE', DebugMode(optimizer='fast_run'))
