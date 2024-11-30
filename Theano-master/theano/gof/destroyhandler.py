"""
Classes and functions for validating graphs that contain view
and inplace operations.

"""
from __future__ import absolute_import, print_function, division

from collections import deque

from six import iteritems

import theano
from . import toolbox
from . import graph
from theano.compat import OrderedDict
from theano.misc.ordered_set import OrderedSet

from .fg import InconsistencyError
from six.moves.queue import Queue


class ProtocolError(Exception):
    """
    Raised when FunctionGraph calls DestroyHandler callbacks in
    an invalid way, for example, pruning or changing a node that has
    never been imported.

    """

    pass


def _contains_cycle(fgraph, orderings):
    """

    Parameters
    ----------
    fgraph
        The FunctionGraph to check for cycles.
    orderings
        Dictionary specifying extra dependencies besides those encoded in
        Variable.owner / Apply.inputs.

        If orderings[my_apply] == dependencies, then my_apply is an Apply
        instance, dependencies is a set of Apply instances, and every member
        of dependencies must be executed before my_apply.

        The dependencies are typically used to prevent
        inplace apply nodes from destroying their input before
        other apply nodes with the same input access it.

    Returns
    -------
    bool
        True if the graph contains a cycle, False otherwise.

    """
    outputs = fgraph.outputs


    assert isinstance(outputs, (tuple, list, deque))



    parent_counts = {}
    node_to_children = {}

    visitable = deque()


    for var in fgraph.variables:

        owner = var.owner
        if owner:
            node_to_children.setdefault(owner, []).append(var)
            parent_counts[var] = 1
        else:
            visitable.append(var)
            parent_counts[var] = 0

    for a_n in fgraph.apply_nodes:
        parents = list(a_n.inputs)
        parents.extend(orderings.get(a_n, []))

        if parents:
            for parent in parents:
                node_to_children.setdefault(parent, []).append(a_n)
            parent_counts[a_n] = len(parents)
        else:
            visitable.append(a_n)
            parent_counts[a_n] = 0



    visited = 0
    while visitable:
        node = visitable.popleft()
        visited += 1
        for client in node_to_children.get(node, []):
            parent_counts[client] -= 1
            if not parent_counts[client]:
                visitable.append(client)

    return visited != len(parent_counts)


def _build_droot_impact(destroy_handler):
    droot = {}   # destroyed view + nonview variables -> foundation
    impact = {}  # destroyed nonview variable -> it + all views of it
    root_destroyer = {}  # root -> destroyer apply

    for app in destroy_handler.destroyers:
        for output_idx, input_idx_list in app.op.destroy_map.items():
            if len(input_idx_list) != 1:
                raise NotImplementedError()
            input_idx = input_idx_list[0]
            input = app.inputs[input_idx]

            view_i = destroy_handler.view_i
            _r = input
            while _r is not None:
                r = _r
                _r = view_i.get(r)
            input_root = r

            if input_root in droot:
                raise InconsistencyError(
                    "Multiple destroyers of %s" % input_root)
            droot[input_root] = input_root
            root_destroyer[input_root] = app

            input_impact = OrderedSet()
            queue = Queue()
            queue.put(input_root)
            while not queue.empty():
                v = queue.get()
                for n in destroy_handler.view_o.get(v, []):
                    input_impact.add(n)
                    queue.put(n)

            for v in input_impact:
                assert v not in droot
                droot[v] = input_root

            impact[input_root] = input_impact
            impact[input_root].add(input_root)

    return droot, impact, root_destroyer


def fast_inplace_check(inputs):
    """
    Return the variables in inputs that are posible candidate for as inputs of
    inplace operation.

    Parameters
    ----------
    inputs : list
        Inputs Variable that you want to use as inplace destination.

    """
    fgraph = inputs[0].fgraph
    Supervisor = theano.compile.function_module.Supervisor
    protected_inputs = [f.protected for f in fgraph._features
                        if isinstance(f, Supervisor)]
    protected_inputs = sum(protected_inputs, [])  # flatten the list
    protected_inputs.extend(fgraph.outputs)

    inputs = [i for i in inputs if
              not isinstance(i, graph.Constant) and
              not fgraph.destroyers(i) and
              i not in protected_inputs]
    return inputs

if 0:
    class DestroyHandler(toolbox.Bookkeeper):
        """
        The DestroyHandler class detects when a graph is impossible to evaluate
        because of aliasing and destructive operations.

        Several data structures are used to do this.

        When an Op uses its view_map property to declare that an output may be
        aliased to an input, then if that output is destroyed, the input is also
        considering to be destroyed. The view_maps of several Ops can feed into
        one another and form a directed graph. The consequence of destroying any
        variable in such a graph is that all variables in the graph must be
        considered to be destroyed, because they could all be refering to the
        same underlying storage. In the current implementation, that graph is a
        tree, and the root of that tree is called the foundation. The `droot`
        property of this class maps from every graph variable to its foundation.
        The `impact` property maps backward from the foundation to all of the
        variables that depend on it. When any variable is destroyed, this class
        marks the foundation of that variable as being destroyed, with the
        `root_destroyer` property.

        """

        droot = {}
        """
        destroyed view + nonview variables -> foundation.

        """
        impact = {}
        """
        destroyed nonview variable -> it + all views of it.

        """
        root_destroyer = {}
        """
        root -> destroyer apply.

        """
        def __init__(self, do_imports_on_attach=True):
            self.fgraph = None
            self.do_imports_on_attach = do_imports_on_attach

        def on_attach(self, fgraph):
            """
            When attaching to a new fgraph, check that
            1) This DestroyHandler wasn't already attached to some fgraph
               (its data structures are only set up to serve one)
            2) The FunctionGraph doesn't already have a DestroyHandler.
               This would result in it validating everything twice, causing
               compilation to be slower.

            TODO: WRITEME: what does this do besides the checks?

            """
            already_there = False
            if self.fgraph not in [None, fgraph]:
                raise Exception("A DestroyHandler instance can only serve"
                                " one FunctionGraph. (Matthew 6:24)")
            for attr in ('destroyers', 'destroy_handler'):
                if hasattr(fgraph, attr):
                    already_there = True

            if already_there:
                raise toolbox.AlreadyThere(
                    "DestroyHandler feature is already present or in"
                    " conflict with another plugin.")


            def get_destroyers_of(r):
                droot, impact, root_destroyer = self.refresh_droot_impact()
                try:
                    return [root_destroyer[droot[r]]]
                except Exception:
                    return []

            fgraph.destroyers = get_destroyers_of
            fgraph.destroy_handler = self

            self.fgraph = fgraph
            self.destroyers = OrderedSet()  # set of Apply instances with non-null destroy_map
            self.view_i = {}  # variable -> variable used in calculation
            self.view_o = {}  # variable -> set of variables that use this one as a direct input
            self.clients = {}  # variable -> apply -> ninputs
            self.stale_droot = True

            if self.do_imports_on_attach:
                toolbox.Bookkeeper.on_attach(self, fgraph)

        def refresh_droot_impact(self):
            if self.stale_droot:
                self.droot, self.impact, self.root_destroyer = _build_droot_impact(self)
                self.stale_droot = False
            return self.droot, self.impact, self.root_destroyer

        def on_detach(self, fgraph):
            if fgraph is not self.fgraph:
                raise Exception("detaching wrong fgraph", fgraph)
            del self.destroyers
            del self.view_i
            del self.view_o
            del self.clients
            del self.stale_droot
            assert self.fgraph.destroyer_handler is self
            delattr(self.fgraph, 'destroyers')
            delattr(self.fgraph, 'destroy_handler')
            self.fgraph = None

        def on_import(self, fgraph, app, reason):
            """
            Add Apply instance to set which must be computed.

            """

            if getattr(app.op, 'destroy_map', {}):
                self.destroyers.add(app)

            for o_idx, i_idx_list in iteritems(getattr(app.op, 'view_map', {})):
                if len(i_idx_list) > 1:
                    raise NotImplementedError(
                        'destroying this output invalidates multiple inputs',
                        (app. op))
                o = app.outputs[o_idx]
                i = app.inputs[i_idx_list[0]]
                self.view_i[o] = i
                self.view_o.setdefault(i, OrderedSet()).add(o)

            for i, input in enumerate(app.inputs):
                self.clients.setdefault(input, {}).setdefault(app, 0)
                self.clients[input][app] += 1

            for i, output in enumerate(app.outputs):
                self.clients.setdefault(output, {})

            self.stale_droot = True

        def on_prune(self, fgraph, app, reason):
            """
            Remove Apply instance from set which must be computed.

            """

            for i, input in enumerate(OrderedSet(app.inputs)):
                del self.clients[input][app]

            if getattr(app.op, 'destroy_map', {}):
                self.destroyers.remove(app)


            for o_idx, i_idx_list in iteritems(getattr(app.op, 'view_map', {})):
                if len(i_idx_list) > 1:
                    raise NotImplementedError()
                o = app.outputs[o_idx]
                i = app.inputs[i_idx_list[0]]

                del self.view_i[o]

                self.view_o[i].remove(o)
                if not self.view_o[i]:
                    del self.view_o[i]

            self.stale_droot = True

        def on_change_input(self, fgraph, app, i, old_r, new_r, reason):
            """
            app.inputs[i] changed from old_r to new_r.

            """
            if app == 'output':
                pass
            else:

                self.clients[old_r][app] -= 1
                if self.clients[old_r][app] == 0:
                    del self.clients[old_r][app]

                self.clients.setdefault(new_r, {}).setdefault(app, 0)
                self.clients[new_r][app] += 1

                for o_idx, i_idx_list in iteritems(getattr(app.op, 'view_map',
                                                           {})):
                    if len(i_idx_list) > 1:
                        raise NotImplementedError()
                    i_idx = i_idx_list[0]
                    output = app.outputs[o_idx]
                    if i_idx == i:
                        if app.inputs[i_idx] is not new_r:
                            raise ProtocolError("wrong new_r on change")

                        self.view_i[output] = new_r

                        self.view_o[old_r].remove(output)
                        if not self.view_o[old_r]:
                            del self.view_o[old_r]

                        self.view_o.setdefault(new_r, OrderedSet()).add(output)

            self.stale_droot = True

        def validate(self, fgraph):
            """
            Return None.

            Raise InconsistencyError when
            a) orderings() raises an error
            b) orderings cannot be topologically sorted.

            """
            if self.destroyers:
                ords = self.orderings(fgraph)

                if _contains_cycle(fgraph, ords):
                    raise InconsistencyError(
                        "Dependency graph contains cycles")
            else:
                pass
            return True

        def orderings(self, fgraph):
            """
            Return orderings induced by destructive operations.

            Raise InconsistencyError when
            a) attempting to destroy indestructable variable, or
            b) attempting to destroy a value multiple times, or
            c) an Apply destroys (illegally) one of its own inputs by aliasing

            """
            rval = OrderedDict()

            if self.destroyers:

                droot, impact, __ignore = self.refresh_droot_impact()

                illegal_destroy = [
                    r for r in droot if
                    getattr(r.tag, 'indestructible', False) or
                    isinstance(r, graph.Constant)]
                if illegal_destroy:
                    raise InconsistencyError(
                        "Attempting to destroy indestructible variables: %s" %
                        illegal_destroy)

                for app in self.destroyers:
                    for output_idx, input_idx_list in iteritems(app.op.destroy_map):
                        destroyed_idx = input_idx_list[0]
                        destroyed_variable = app.inputs[destroyed_idx]
                        root = droot[destroyed_variable]
                        root_impact = impact[root]

                        tolerate_same = getattr(app.op,
                                                'destroyhandler_tolerate_same',
                                                [])
                        assert isinstance(tolerate_same, list)
                        tolerated = OrderedSet(idx1 for idx0, idx1 in
                                               tolerate_same
                                               if idx0 == destroyed_idx)
                        tolerated.add(destroyed_idx)
                        tolerate_aliased = getattr(
                            app.op, 'destroyhandler_tolerate_aliased', [])
                        assert isinstance(tolerate_aliased, list)
                        ignored = OrderedSet(idx1 for idx0, idx1
                                             in tolerate_aliased
                                             if idx0 == destroyed_idx)
                        for i, input in enumerate(app.inputs):
                            if i in ignored:
                                continue
                            if input in root_impact \
                                    and (i not in tolerated or input is not destroyed_variable):
                                raise InconsistencyError("Input aliasing: %s (%i, %i)"
                                                         % (app, destroyed_idx, i))

                        root_clients = OrderedSet()
                        for r in root_impact:
                            assert not [a for a, c in
                                        iteritems(self.clients[r]) if not c]
                            root_clients.update([a for a, c in
                                                 iteritems(self.clients[r])
                                                 if c])
                        root_clients.remove(app)
                        if root_clients:
                            rval[app] = root_clients

            return rval


class DestroyHandler(toolbox.Bookkeeper):  # noqa
    """
    The DestroyHandler class detects when a graph is impossible to evaluate
    because of aliasing and destructive operations.

    Several data structures are used to do this.

    An Op can use its view_map property to declare that an output may be
    aliased to an input. If that output is destroyed, the input is also
    considered to be destroyed. The view_maps of several Ops can feed into
    one another and form a directed graph. The consequence of destroying any
    variable in such a graph is that all variables in the graph must be
    considered to be destroyed, because they could all be refering to the
    same underlying storage.

    In the current implementation, that graph is a tree, and the root of that
    tree is called the foundation.

    TODO: why "in the current implementation" ? is there another implementation
          planned?
    TODO: why is the graph a tree? isn't it possible that one variable could
          be aliased to many variables? for example, don't switch and ifelse
          have to do this?

    The original DestroyHandler (if 0'ed out above) computed several data
    structures from scratch each time it was asked to validate the graph.
    Because this happens potentially thousands of times and each graph to
    validate is extremely similar to the previous one, computing the
    data structures from scratch repeatedly was wasteful and resulted in
    high compile times for large graphs.

    This implementation computes the data structures once at initialization
    and then incrementally updates them.

    It is a work in progress. The following data structures have been
    converted to use the incremental strategy:
        <none>

    The following data structures remain to be converted:
        <unknown>

    """
    pickle_rm_attr = ["destroyers"]

    def __init__(self, do_imports_on_attach=True):
        self.fgraph = None
        self.do_imports_on_attach = do_imports_on_attach

        """
        Maps every variable in the graph to its "foundation" (deepest
        ancestor in view chain).
        TODO: change name to var_to_vroot.

        """
        self.droot = OrderedDict()

        """
        Maps a variable to all variables that are indirect or direct views of it
        (including itself) essentially the inverse of droot.
        TODO: do all variables appear in this dict, or only those that are
              foundations?
        TODO: do only destroyed variables go in here? one old docstring said so.
        TODO: rename to x_to_views after reverse engineering what x is

        """
        self.impact = OrderedDict()

        """
        If a var is destroyed, then this dict will map
        droot[var] to the apply node that destroyed var
        TODO: rename to vroot_to_destroyer

        """
        self.root_destroyer = OrderedDict()

    def on_attach(self, fgraph):
        """
        When attaching to a new fgraph, check that
            1) This DestroyHandler wasn't already attached to some fgraph
               (its data structures are only set up to serve one).
            2) The FunctionGraph doesn't already have a DestroyHandler.
               This would result in it validating everything twice, causing
               compilation to be slower.

        Give the FunctionGraph instance:
            1) A new method "destroyers(var)"
               TODO: what does this do exactly?
            2) A new attribute, "destroy_handler"
        TODO: WRITEME: what does this do besides the checks?

        """

        already_there = False
        if self.fgraph is fgraph:
            already_there = True
        if self.fgraph is not None:
            raise Exception(
                "A DestroyHandler instance can only serve one"
                " FunctionGraph. (Matthew 6:24)")
        for attr in ('destroyers', 'destroy_handler'):
            if hasattr(fgraph, attr):
                already_there = True

        if already_there:
            raise toolbox.AlreadyThere(
                "DestroyHandler feature is already present"
                " or in conflict with another plugin.")

        self.unpickle(fgraph)
        fgraph.destroy_handler = self

        self.fgraph = fgraph
        self.destroyers = OrderedSet()  # set of Apply instances with non-null destroy_map
        self.view_i = OrderedDict()  # variable -> variable used in calculation
        self.view_o = OrderedDict()  # variable -> set of variables that use this one as a direct input
        self.clients = OrderedDict()  # variable -> apply -> ninputs
        self.stale_droot = True

        self.debug_all_apps = OrderedSet()
        if self.do_imports_on_attach:
            toolbox.Bookkeeper.on_attach(self, fgraph)

    def unpickle(self, fgraph):
        def get_destroyers_of(r):
            droot, impact, root_destroyer = self.refresh_droot_impact()
            try:
                return [root_destroyer[droot[r]]]
            except Exception:
                return []
        fgraph.destroyers = get_destroyers_of

    def refresh_droot_impact(self):
        """
        Makes sure self.droot, self.impact, and self.root_destroyer are up to
        date, and returns them (see docstrings for these properties above).

        """
        if self.stale_droot:
            self.droot, self.impact, self.root_destroyer =\
                _build_droot_impact(self)
            self.stale_droot = False
        return self.droot, self.impact, self.root_destroyer

    def on_detach(self, fgraph):
        if fgraph is not self.fgraph:
            raise Exception("detaching wrong fgraph", fgraph)
        del self.destroyers
        del self.view_i
        del self.view_o
        del self.clients
        del self.stale_droot
        assert self.fgraph.destroyer_handler is self
        delattr(self.fgraph, 'destroyers')
        delattr(self.fgraph, 'destroy_handler')
        self.fgraph = None

    def on_import(self, fgraph, app, reason):
        """
        Add Apply instance to set which must be computed.

        """

        if app in self.debug_all_apps:
            raise ProtocolError("double import")
        self.debug_all_apps.add(app)

        if getattr(app.op, 'destroy_map', {}):
            self.destroyers.add(app)

        for o_idx, i_idx_list in iteritems(getattr(app.op, 'view_map', {})):
            if len(i_idx_list) > 1:
                raise NotImplementedError(
                    'destroying this output invalidates multiple inputs',
                    (app. op))
            o = app.outputs[o_idx]
            i = app.inputs[i_idx_list[0]]
            self.view_i[o] = i
            self.view_o.setdefault(i, OrderedSet()).add(o)

        for i, input in enumerate(app.inputs):
            self.clients.setdefault(input, OrderedDict()).setdefault(app, 0)
            self.clients[input][app] += 1

        for i, output in enumerate(app.outputs):
            self.clients.setdefault(output, OrderedDict())

        self.stale_droot = True

    def on_prune(self, fgraph, app, reason):
        """
        Remove Apply instance from set which must be computed.

        """
        if app not in self.debug_all_apps:
            raise ProtocolError("prune without import")
        self.debug_all_apps.remove(app)

        for i, input in enumerate(OrderedSet(app.inputs)):
            del self.clients[input][app]

        if getattr(app.op, 'destroy_map', OrderedDict()):
            self.destroyers.remove(app)


        for o_idx, i_idx_list in iteritems(getattr(app.op, 'view_map',
                                                   OrderedDict())):
            if len(i_idx_list) > 1:
                raise NotImplementedError()
            o = app.outputs[o_idx]
            i = app.inputs[i_idx_list[0]]

            del self.view_i[o]

            self.view_o[i].remove(o)
            if not self.view_o[i]:
                del self.view_o[i]

        self.stale_droot = True

    def on_change_input(self, fgraph, app, i, old_r, new_r, reason):
        """
        app.inputs[i] changed from old_r to new_r.

        """
        if app == 'output':
            pass
        else:
            if app not in self.debug_all_apps:
                raise ProtocolError("change without import")

            self.clients[old_r][app] -= 1
            if self.clients[old_r][app] == 0:
                del self.clients[old_r][app]

            self.clients.setdefault(new_r, OrderedDict()).setdefault(app, 0)
            self.clients[new_r][app] += 1

            for o_idx, i_idx_list in iteritems(getattr(app.op, 'view_map',
                                                       OrderedDict())):
                if len(i_idx_list) > 1:
                    raise NotImplementedError()
                i_idx = i_idx_list[0]
                output = app.outputs[o_idx]
                if i_idx == i:
                    if app.inputs[i_idx] is not new_r:
                        raise ProtocolError("wrong new_r on change")

                    self.view_i[output] = new_r

                    self.view_o[old_r].remove(output)
                    if not self.view_o[old_r]:
                        del self.view_o[old_r]

                    self.view_o.setdefault(new_r, OrderedSet()).add(output)

        self.stale_droot = True

    def validate(self, fgraph):
        """
        Return None.

        Raise InconsistencyError when
        a) orderings() raises an error
        b) orderings cannot be topologically sorted.

        """
        if self.destroyers:
            ords = self.orderings(fgraph)

            if _contains_cycle(fgraph, ords):
                raise InconsistencyError("Dependency graph contains cycles")
        else:

            pass
        return True

    def orderings(self, fgraph):
        """
        Return orderings induced by destructive operations.

        Raise InconsistencyError when
        a) attempting to destroy indestructable variable, or
        b) attempting to destroy a value multiple times, or
        c) an Apply destroys (illegally) one of its own inputs by aliasing

        """
        rval = OrderedDict()

        if self.destroyers:

            droot, impact, __ignore = self.refresh_droot_impact()

            illegal_destroy = [r for r in droot if
                               getattr(r.tag, 'indestructible', False) or
                               isinstance(r, graph.Constant)]
            if illegal_destroy:
                raise InconsistencyError(
                    "Attempting to destroy indestructible variables: %s" %
                    illegal_destroy)

            for app in self.destroyers:
                for output_idx, input_idx_list in iteritems(app.op.destroy_map):
                    destroyed_idx = input_idx_list[0]
                    destroyed_variable = app.inputs[destroyed_idx]
                    root = droot[destroyed_variable]
                    root_impact = impact[root]

                    tolerate_same = getattr(app.op,
                                            'destroyhandler_tolerate_same', [])
                    assert isinstance(tolerate_same, list)
                    tolerated = OrderedSet(idx1 for idx0, idx1 in tolerate_same
                                           if idx0 == destroyed_idx)
                    tolerated.add(destroyed_idx)
                    tolerate_aliased = getattr(
                        app.op, 'destroyhandler_tolerate_aliased', [])
                    assert isinstance(tolerate_aliased, list)
                    ignored = OrderedSet(idx1 for idx0, idx1 in tolerate_aliased
                                         if idx0 == destroyed_idx)
                    for i, input in enumerate(app.inputs):
                        if i in ignored:
                            continue
                        if input in root_impact \
                                and (i not in tolerated or
                                     input is not destroyed_variable):
                            raise InconsistencyError("Input aliasing: %s (%i, %i)"
                                                     % (app, destroyed_idx, i))

                    root_clients = OrderedSet()
                    for r in root_impact:
                        assert not [a for a, c in self.clients[r].items() if not c]
                        root_clients.update([a for a, c in self.clients[r].items() if c])
                    root_clients.remove(app)
                    if root_clients:
                        rval[app] = root_clients

        return rval
