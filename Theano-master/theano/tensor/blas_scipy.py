"""
Implementations of BLAS Ops based on scipy's BLAS bindings.
"""
from __future__ import absolute_import, print_function, division
import numpy

from theano.tensor.blas import Ger, ger, ger_destructive, have_fblas
from theano.tensor.blas import blas_optdb, optdb, local_optimizer

from theano.tensor.opt import in2out


if have_fblas:
    from theano.tensor.blas import fblas
    _blas_ger_fns = {
        numpy.dtype('float32'): fblas.sger,
        numpy.dtype('float64'): fblas.dger,
        numpy.dtype('complex64'): fblas.cgeru,
        numpy.dtype('complex128'): fblas.zgeru,
    }


class ScipyGer(Ger):

    def make_thunk(self, node, storage_map, compute_map, no_recycling):

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_output_compute = [compute_map[r] for r in node.outputs]

        cA, calpha, cx, cy = node_input_storage
        cZ, = node_output_storage
        local_ger = _blas_ger_fns[numpy.dtype(node.inputs[0].type.dtype)]

        def rval():
            A = cA[0]
            if A.size == 0:
                if not self.destructive:
                    A = A.copy()
            elif A.flags['C_CONTIGUOUS']:
                A = local_ger(calpha[0], cy[0], cx[0], a=A.T,
                              overwrite_a=int(self.destructive)).T
            else:
                A = local_ger(calpha[0], cx[0], cy[0], a=A,
                              overwrite_a=int(self.destructive))
            cZ[0] = A
            for o in node_output_compute:
                o[0] = True

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.lazy = False
        return rval

scipy_ger_no_inplace = ScipyGer(False)
scipy_ger_inplace = ScipyGer(True)


@local_optimizer([ger, ger_destructive])
def use_scipy_ger(node):
    if node.op == ger:
        return [scipy_ger_no_inplace(*node.inputs)]


@local_optimizer([scipy_ger_no_inplace])
def make_ger_destructive(node):
    if node.op == scipy_ger_no_inplace:
        return [scipy_ger_inplace(*node.inputs)]

use_scipy_blas = in2out(use_scipy_ger)
make_scipy_blas_destructive = in2out(make_ger_destructive)

if have_fblas:
    blas_optdb.register('scipy_blas',
                        use_scipy_blas,
                        100, 'fast_run')

    optdb.register('make_scipy_blas_destructive',
                   make_scipy_blas_destructive,
                   70.0, 'fast_run', 'inplace')
