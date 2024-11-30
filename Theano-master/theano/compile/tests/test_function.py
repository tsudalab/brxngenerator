from __future__ import absolute_import, print_function, division
import six.moves.cPickle as pickle
import os
import shutil
import tempfile
import unittest

import numpy

import theano
from theano.compile.io import In


def test_function_dump():
    v = theano.tensor.vector()
    fct1 = theano.function([v], v + 1)

    try:
        tmpdir = tempfile.mkdtemp()
        fname = os.path.join(tmpdir, 'test_function_dump.pkl')
        theano.function_dump(fname, [v], v + 1)
        with open(fname, 'rb') as f:
            l = pickle.load(f)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

    fct2 = theano.function(**l)
    x = [1, 2, 3]
    assert numpy.allclose(fct1(x), fct2(x))


class TestFunctionIn(unittest.TestCase):

    def test_in_strict(self):

        a = theano.tensor.dvector()
        b = theano.shared(7)
        out = a + b

        f = theano.function([In(a, strict=False)], out)
        f(numpy.random.rand(8))
        f(numpy.array([1, 2, 3, 4], dtype='int32'))

        f = theano.function([In(a, strict=True)], out)
        try:
            f(numpy.array([1, 2, 3, 4], dtype='int32'))
        except TypeError:
            pass

    def test_explicit_shared_input(self):
        a = theano.shared(1.0)
        self.assertRaises(TypeError, theano.function, [a], a + 1)

    def test_in_shared_variable(self):
        a = theano.shared(1.0)
        a_wrapped = In(a, update=a + 1)
        self.assertRaises(TypeError, theano.function, [a_wrapped])

    def test_in_mutable(self):
        a = theano.tensor.dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        f = theano.function([In(a, mutable=True)], a_out, mode='FAST_RUN')
        aval = numpy.random.rand(10)
        aval2 = aval.copy()
        assert numpy.all(f(aval) == (aval2 * 2))
        assert not numpy.all(aval == aval2)

        f = theano.function([In(a, mutable=False)], a_out, mode='FAST_RUN')
        aval = numpy.random.rand(10)
        aval2 = aval.copy()
        assert numpy.all(f(aval) == (aval2 * 2))
        assert numpy.all(aval == aval2)

    def test_in_update(self):
        a = theano.tensor.dscalar('a')
        f = theano.function([In(a, value=0.0, update=a + 1)], a,
                            mode='FAST_RUN')

        assert f() == 0.0
        assert f() == 1.0
        assert f() == 2.0

    def test_in_update_wrong_dtype(self):
        a = theano.tensor.dscalar('a')
        b = theano.tensor.dvector('b')
        self.assertRaises(TypeError, In, a, update=b)

    def test_in_update_shared(self):
        shared_var = theano.shared(1.0)
        a = theano.tensor.dscalar('a')
        a_wrapped = In(a, value=0.0, update=shared_var)
        f = theano.function([a_wrapped], [], updates={shared_var: a},
                            mode='FAST_RUN')

        for i in range(5):
            f()
            assert numpy.allclose(shared_var.get_value(), i % 2)

    def test_in_allow_downcast_int(self):
        a = theano.tensor.wvector('a')  # int16
        b = theano.tensor.bvector('b')  # int8
        c = theano.tensor.bscalar('c')  # int8
        f = theano.function([In(a, allow_downcast=True),
                             In(b, allow_downcast=False),
                             In(c, allow_downcast=None)],
                            (a + b + c))

        assert numpy.all(f([3], [6], 1) == 10)

        self.assertRaises(TypeError, f, [3], numpy.array([6], dtype='int16'),
                          1)

        assert numpy.all(f([2 ** 20], numpy.ones(1, dtype='int8'), 1) == 2)

        self.assertRaises(TypeError, f, [3], [312], 1)

        self.assertRaises(TypeError, f, [3], [6], 806)

    def test_in_allow_downcast_floatX(self):
        a = theano.tensor.fscalar('a')
        b = theano.tensor.fscalar('b')
        c = theano.tensor.fscalar('c')

        f = theano.function([In(a, allow_downcast=True),
                             In(b, allow_downcast=False),
                             In(c, allow_downcast=None)],
                            (a + b + c))

        assert numpy.all(f(0, 0, 0) == 0)

        assert numpy.allclose(f(0.1, 0, 0), 0.1)

        self.assertRaises(TypeError, f, 0, 0.1, 0)

        if theano.config.floatX == 'float32':
            assert numpy.allclose(f(0, 0, 0.1), 0.1)
        else:
            self.assertRaises(TypeError, f, 0, 0, 0.1)

    def test_in_allow_downcast_vector_floatX(self):
        a = theano.tensor.fvector('a')
        b = theano.tensor.fvector('b')
        c = theano.tensor.fvector('c')

        f = theano.function([In(a, allow_downcast=True),
                             In(b, allow_downcast=False),
                             In(c, allow_downcast=None)],
                            (a + b + c))

        z = [0]
        assert numpy.all(f(z, z, z) == 0)

        assert numpy.allclose(f([0.1], z, z), 0.1)

        self.assertRaises(TypeError, f, z, [0.1], z)

        self.assertRaises(TypeError, f, z, z, [0.1])
