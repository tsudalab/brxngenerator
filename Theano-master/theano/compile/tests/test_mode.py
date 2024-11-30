from __future__ import absolute_import, print_function, division
import theano
from theano.compile.mode import Mode, AddFeatureOptimizer
from theano.gof.toolbox import NoOutputFromInplace
import theano.tensor as T


def test_no_output_from_implace():

    x = T.matrix()
    y = T.matrix()
    a = T.dot(x, y)
    b = T.tanh(a)

    fct_no_opt = theano.function([x, y], b, mode="FAST_RUN")
    op = fct_no_opt.maker.fgraph.outputs[0].owner.op
    assert (hasattr(op, 'destroy_map') and 0 in op.destroy_map)

    opt = AddFeatureOptimizer(NoOutputFromInplace())
    mode_opt = Mode(linker="cvm", optimizer="fast_run").register((opt, 49.9))

    fct_opt = theano.function([x, y], b, mode=mode_opt)
    op = fct_opt.maker.fgraph.outputs[0].owner.op
    assert (not hasattr(op, 'destroy_map') or 0 not in op.destroy_map)


def test_including():
    mode = theano.Mode(optimizer='merge')
    mode.including('fast_compile')
