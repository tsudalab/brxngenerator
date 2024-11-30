from __future__ import absolute_import, print_function, division
import unittest
from theano.tests import unittest_tools as utt

import numpy as np

import theano
from theano import tensor

from theano.tensor.sort import sort, SortOp
from theano.tensor.sort import argsort, ArgSortOp


class test_sort(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(seed=utt.fetch_seed())
        self.m_val = self.rng.rand(3, 2)
        self.v_val = self.rng.rand(4)

    def test1(self):
        a = tensor.dmatrix()
        w = sort(a)
        f = theano.function([a], w)
        assert np.allclose(f(self.m_val), np.sort(self.m_val))

    def test2(self):
        a = tensor.dmatrix()
        axis = tensor.scalar()
        w = sort(a, axis)
        f = theano.function([a, axis], w)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            assert np.allclose(gv, gt)

    def test3(self):
        a = tensor.dvector()
        w2 = sort(a)
        f = theano.function([a], w2)
        gv = f(self.v_val)
        gt = np.sort(self.v_val)
        assert np.allclose(gv, gt)

    def test4(self):
        a = tensor.dmatrix()
        axis = tensor.scalar()
        l = sort(a, axis, "mergesort")
        f = theano.function([a, axis], l)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            assert np.allclose(gv, gt)

    def test5(self):
        a1 = SortOp("mergesort", [])
        a2 = SortOp("quicksort", [])

        assert a1 != a2
        assert a1 == SortOp("mergesort", [])
        assert a2 == SortOp("quicksort", [])

    def test_None(self):
        a = tensor.dmatrix()
        l = sort(a, None)
        f = theano.function([a], l)
        gv = f(self.m_val)
        gt = np.sort(self.m_val, None)
        assert np.allclose(gv, gt)

    def test_grad_vector(self):
        a = theano.tensor.vector()
        data = np.random.rand(10).astype(theano.config.floatX)
        utt.verify_grad(sort, [data])

    def test_grad_none_axis(self):
        data = np.random.rand(10).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        utt.verify_grad(lambda x: sort(x, 0), [data])

        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])

    def test_grad_negative_axis(self):
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])

        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])

        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -4), [data])

    def test_grad_nonnegative_axis(self):
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])

        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])

        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 3), [data])


class TensorInferShapeTester(utt.InferShapeTester):
    def test_sort(self):
        x = tensor.matrix()
        self._compile_and_check(
                [x],
                [sort(x)],
                [np.random.randn(10, 40).astype(theano.config.floatX)],
                SortOp)
        self._compile_and_check(
                [x],
                [sort(x, axis=None)],
                [np.random.randn(10, 40).astype(theano.config.floatX)],
                SortOp)


def test_argsort():
    rng = np.random.RandomState(seed=utt.fetch_seed())
    m_val = rng.rand(3, 2)
    v_val = rng.rand(4)

    a = tensor.dmatrix()
    w = argsort(a)
    f = theano.function([a], w)
    gv = f(m_val)
    gt = np.argsort(m_val)
    assert np.allclose(gv, gt)

    a = tensor.dmatrix()
    axis = tensor.lscalar()
    w = argsort(a, axis)
    f = theano.function([a, axis], w)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        assert np.allclose(gv, gt)

    a = tensor.dvector()
    w2 = argsort(a)
    f = theano.function([a], w2)
    gv = f(v_val)
    gt = np.argsort(v_val)
    assert np.allclose(gv, gt)

    a = tensor.dmatrix()
    axis = tensor.lscalar()
    l = argsort(a, axis, "mergesort")
    f = theano.function([a, axis], l)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        assert np.allclose(gv, gt)

    a = tensor.dmatrix()
    axis = tensor.lscalar()
    a1 = ArgSortOp("mergesort", [])
    a2 = ArgSortOp("quicksort", [])
    assert a1 != a2
    assert a1 == ArgSortOp("mergesort", [])
    assert a2 == ArgSortOp("quicksort", [])

    a = tensor.dmatrix()
    w2 = argsort(a, None)
    f = theano.function([a], w2)
    gv = f(m_val)
    gt = np.argsort(m_val, None)
    assert np.allclose(gv, gt)


def test_argsort_grad():
    data = np.random.rand(2, 3).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-1), [data])

    data = np.random.rand(2, 3, 4, 5).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-3), [data])

    data = np.random.rand(2, 3, 3).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=2), [data])
