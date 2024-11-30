from __future__ import absolute_import, print_function, division
import numpy
import six.moves.cPickle as pickle
from six.moves import xrange
import theano
from theano import tensor as T
import time


def test_no_reuse():
    x = T.lvector()
    y = T.lvector()
    f = theano.function([x, y], x + y)

    f(numpy.ones(10, dtype='int64'), numpy.ones(10, dtype='int64'))

    try:
        f(numpy.ones(10))
    except TypeError:
        return
    assert not 'should not get here'


def test_gc_never_pickles_temporaries():
    x = T.dvector()

    for i in xrange(2):  # TODO: 30 causes like LONG compilation due to MERGE
        if i:
            r = r + r/10
        else:
            r = x

    optimizer = None
    optimizer = 'fast_run'

    for f_linker, g_linker in [
            (theano.PerformLinker(allow_gc=True),
             theano.PerformLinker(allow_gc=False)),
            (theano.OpWiseCLinker(allow_gc=True),
             theano.OpWiseCLinker(allow_gc=False))]:



        f = theano.function([x], r, mode=theano.Mode(optimizer=optimizer,
                                                     linker=f_linker))
        g = theano.function([x], r, mode=theano.Mode(optimizer=optimizer,
                                                     linker=g_linker))

        len_pre_f = len(pickle.dumps(f))
        len_pre_g = len(pickle.dumps(g))


        def a(fn):
            return len(pickle.dumps(fn.maker))
        assert a(f) == a(f)  # some sanity checks on the pickling mechanism
        assert a(g) == a(g)  # some sanity checks on the pickling mechanism

        def b(fn):
            return len(
                pickle.dumps(
                    theano.compile.function_module._pickle_Function(
                        fn)))
        assert b(f) == b(f)  # some sanity checks on the pickling mechanism

        def c(fn):
            return len(pickle.dumps(fn))
        assert c(f) == c(f)  # some sanity checks on the pickling mechanism
        assert c(g) == c(g)  # some sanity checks on the pickling mechanism

        f(numpy.ones(100, dtype='float64'))
        g(numpy.ones(100, dtype='float64'))

        post_f = pickle.dumps(f)
        post_g = pickle.dumps(g)
        len_post_f = len(post_f)
        len_post_g = len(post_g)

        assert len_pre_f == len_post_f

        assert abs(len_post_f - len_post_g) < 256, (
            f_linker, len_post_f, len_post_g)


def test_merge_opt_runtime():
    """In the original merge optimization, the following graph took
    like caused the MERGE optimizer to exhibit really bad performance
    (quadratic? exponential?)

    Ironically, there is actually no merging to do in this graph.

    """
    x = T.dvector()
    for i in xrange(50):
        if i:
            r = r + r/10
        else:
            r = x
    t = time.time()
    f = theano.function([x], r, mode='FAST_COMPILE')
    dt = time.time() - t

    assert dt < 5.0, dt
