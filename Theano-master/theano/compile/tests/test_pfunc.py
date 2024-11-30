from __future__ import absolute_import, print_function, division
import unittest

from nose.plugins.skip import SkipTest
import numpy

import theano
from theano.tensor import dmatrix, iscalar, lscalar, dmatrices
from theano import tensor

from theano.compile import In
from theano.compile import pfunc
from theano.compile import shared
from theano.compile import config


def data_of(s):
    """Return the raw value of a shared variable"""
    return s.container.storage[0]


class Test_pfunc(unittest.TestCase):

    def test_doc(self):
        """Ensure the code given in pfunc.txt works as expected"""

        a = lscalar()
        b = shared(1)
        f1 = pfunc([a], (a + b))
        f2 = pfunc([In(a, value=44)], a + b, updates={b: b + 1})
        self.assertTrue(b.get_value() == 1)
        self.assertTrue(f1(3) == 4)
        self.assertTrue(f2(3) == 4)
        self.assertTrue(b.get_value() == 2)
        self.assertTrue(f1(3) == 5)
        b.set_value(0)
        self.assertTrue(f1(3) == 3)

        a = tensor.lscalar()
        b = shared(7)
        f1 = pfunc([a], a + b)
        f2 = pfunc([a], a * b)
        self.assertTrue(f1(5) == 12)
        b.set_value(8)
        self.assertTrue(f1(5) == 13)
        self.assertTrue(f2(4) == 32)

    def test_shared(self):

        w = shared(numpy.random.rand(2, 2), 'w')
        wval = w.get_value(borrow=False)

        x = dmatrix()
        out1 = w + x
        out2 = w * x
        f1 = pfunc([x], [out1])
        f2 = pfunc([x], [out2])
        xval = numpy.random.rand(2, 2)
        assert numpy.all(f1(xval) == xval + wval)
        assert numpy.all(f2(xval) == xval * wval)

        f3 = pfunc([x], out1, updates=[(w, (w - 1))])
        assert numpy.all(f3(xval) == xval + wval)
        assert numpy.all(f1(xval) == xval + (wval - 1))

        w.set_value(w.get_value(borrow=True) * 10, borrow=True)
        assert numpy.all(f1(xval) == xval + w.get_value(borrow=True))

    def test_no_shared_as_input(self):
        """Test that shared variables cannot be used as function inputs."""
        w_init = numpy.random.rand(2, 2)
        w = shared(w_init.copy(), 'w')
        try:
            pfunc([w], theano.tensor.sum(w * w))
            assert False
        except TypeError as e:
            msg = 'Cannot use a shared variable (w) as explicit input'
            if str(e).find(msg) < 0:
                raise

    def test_default_container(self):

        rng = numpy.random.RandomState(1827)
        w_init = rng.rand(5)
        w = shared(w_init.copy(), 'w')
        reg = theano.tensor.sum(w * w)
        f = pfunc([], reg)

        assert f() == numpy.sum(w_init * w_init)
        w.set_value(w.get_value(borrow=True) + 1.0, borrow=True)
        assert f() == numpy.sum((w_init + 1) ** 2)

    def test_default_scalar_container(self):
        x = shared(0.0, 'x')
        f = pfunc([], x)
        assert f() == 0
        x.set_value(x.get_value(borrow=True) + 1, borrow=True)
        assert f() == 1

    def test_param_strict(self):

        a = tensor.dvector()
        b = shared(7)
        out = a + b

        f = pfunc([In(a, strict=False)], [out])
        f(numpy.random.rand(8))
        f(numpy.array([1, 2, 3, 4], dtype='int32'))

        f = pfunc([In(a, strict=True)], [out])
        try:
            f(numpy.array([1, 2, 3, 4], dtype='int32'))
        except TypeError:
            pass

    def test_param_mutable(self):
        a = tensor.dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        fip = pfunc([In(a, mutable=True)], [a_out], mode='FAST_RUN')
        aval = numpy.random.rand(10)
        aval2 = aval.copy()
        assert numpy.all(fip(aval) == (aval2 * 2))
        assert not numpy.all(aval == aval2)

        f = pfunc([In(a, mutable=False)], [a_out], mode='FAST_RUN')
        aval = numpy.random.rand(10)
        aval2 = aval.copy()
        assert numpy.all(f(aval) == (aval2 * 2))
        assert numpy.all(aval == aval2)

    def test_shared_mutable(self):
        bval = numpy.arange(5)
        b = shared(bval)
        b_out = b * 2

        assert b.get_value(borrow=True) is not bval
        bval = data_of(b)

        f = pfunc([], [b_out], mode='FAST_RUN')
        assert (f() == numpy.arange(5) * 2).all()
        assert numpy.all(b.get_value(borrow=True) == numpy.arange(5))

        f = pfunc([], [b_out], updates=[(b, b_out)], mode='FAST_RUN')
        assert (f() == (numpy.arange(5) * 2)).all()
        assert (b.get_value(borrow=True) == (numpy.arange(5) * 2)).all()
        assert (bval == (numpy.arange(5) * 2)).all()  # because of mutable=True

        bval = numpy.arange(5)
        b.set_value(bval, borrow=True)
        bval = data_of(b)
        f = pfunc([], [b_out], updates=[(b, (b_out + 3))], mode='FAST_RUN')
        assert (f() == (numpy.arange(5) * 2)).all()
        assert (b.get_value(borrow=True) == ((numpy.arange(5) * 2) + 3)).all()
        assert not (bval == numpy.arange(5)).all()
        assert not (bval == b.get_value(borrow=True)).all()

    def test_param_allow_downcast_int(self):
        a = tensor.wvector('a')  # int16
        b = tensor.bvector('b')  # int8
        c = tensor.bscalar('c')  # int8
        f = pfunc([In(a, allow_downcast=True),
                   In(b, allow_downcast=False),
                   In(c, allow_downcast=None)],
                  (a + b + c))

        assert numpy.all(f([3], [6], 1) == 10)

        self.assertRaises(TypeError, f,
                          [3], numpy.array([6], dtype='int16'), 1)

        assert numpy.all(f([2 ** 20], numpy.ones(1, dtype='int8'), 1) == 2)

        self.assertRaises(TypeError, f, [3], [312], 1)

        self.assertRaises(TypeError, f, [3], [6], 806)

    def test_param_allow_downcast_floatX(self):
        a = tensor.fscalar('a')
        b = tensor.fscalar('b')
        c = tensor.fscalar('c')

        f = pfunc([In(a, allow_downcast=True),
                   In(b, allow_downcast=False),
                   In(c, allow_downcast=None)],
                  (a + b + c))

        assert numpy.all(f(0, 0, 0) == 0)

        assert numpy.allclose(f(0.1, 0, 0), 0.1)

        self.assertRaises(TypeError, f, 0, 0.1, 0)

        if config.floatX == 'float32':
            assert numpy.allclose(f(0, 0, 0.1), 0.1)
        else:
            self.assertRaises(TypeError, f, 0, 0, 0.1)

    def test_param_allow_downcast_vector_floatX(self):
        a = tensor.fvector('a')
        b = tensor.fvector('b')
        c = tensor.fvector('c')

        f = pfunc([In(a, allow_downcast=True),
                   In(b, allow_downcast=False),
                   In(c, allow_downcast=None)],
                  (a + b + c))

        z = [0]
        assert numpy.all(f(z, z, z) == 0)

        assert numpy.allclose(f([0.1], z, z), 0.1)

        self.assertRaises(TypeError, f, z, [0.1], z)

        self.assertRaises(TypeError, f, z, z, [0.1])

    def test_allow_input_downcast_int(self):
        a = tensor.wvector('a')  # int16
        b = tensor.bvector('b')  # int8
        c = tensor.bscalar('c')  # int8

        f = pfunc([a, b, c], (a + b + c), allow_input_downcast=True)
        assert f([2 ** 20], [1], 0) == 1
        assert f([3], [312], 0) == 59
        assert f([3], [1], 806) == 42

        g = pfunc([a, b, c], (a + b + c), allow_input_downcast=False)
        assert numpy.all(g([3], [6], 0) == 9)

        self.assertRaises(TypeError, g,
                          [3], numpy.array([6], dtype='int16'), 0)

        self.assertRaises(TypeError, g, [3], [312], 0)

        h = pfunc([a, b, c], (a + b + c))  # Default: allow_input_downcast=None
        assert numpy.all(h([3], [6], 0) == 9)
        self.assertRaises(TypeError, h,
                          [3], numpy.array([6], dtype='int16'), 0)
        self.assertRaises(TypeError, h, [3], [312], 0)

    def test_allow_downcast_floatX(self):
        a = tensor.fscalar('a')
        b = tensor.fvector('b')

        f = pfunc([a, b], (a + b), allow_input_downcast=True)
        g = pfunc([a, b], (a + b), allow_input_downcast=False)
        h = pfunc([a, b], (a + b), allow_input_downcast=None)

        assert numpy.all(f(0, [0]) == 0)
        assert numpy.all(g(0, [0]) == 0)
        assert numpy.all(h(0, [0]) == 0)

        assert numpy.allclose(f(0, [0.1]), 0.1)
        self.assertRaises(TypeError, g, 0, [0.1])
        self.assertRaises(TypeError, h, 0, [0.1])

        assert numpy.allclose(f(0.1, [0]), 0.1)
        self.assertRaises(TypeError, g, 0.1, [0])
        if config.floatX == 'float32':
            assert numpy.allclose(h(0.1, [0]), 0.1)
        else:
            self.assertRaises(TypeError, h, 0.1, [0])

    def test_update(self):
        """Test update mechanism in different settings."""

        x = shared(0)
        assign = pfunc([], [], updates={x: 3})
        assign()
        self.assertTrue(x.get_value() == 3)

        x.set_value(0)
        inc = pfunc([], [], updates={x: x + 1})
        inc()
        self.assertTrue(x.get_value() == 1)

        x.set_value(-1)
        y = shared(2)
        inc_by_y = pfunc([], [], updates={x: x + y})
        inc_by_y()
        self.assertTrue(x.get_value() == 1)

    def test_update_err_broadcast(self):
        data = numpy.random.rand(10, 10).astype('float32')
        output_var = shared(name="output", value=data)

        self.assertRaises(TypeError, theano.function, inputs=[], outputs=[],
                          updates={output_var: output_var.sum().dimshuffle('x', 'x')})

    def test_duplicate_updates(self):
        x, y = dmatrices('x', 'y')
        z = shared(numpy.ones((2, 3)))
        self.assertRaises(ValueError, theano.function, [x, y], [z],
                          updates=[(z, (z + x + y)), (z, (z - x))])

    def test_givens(self):
        x = shared(0)
        assign = pfunc([], x, givens={x: 3})
        assert assign() == 3
        assert x.get_value(borrow=True) == 0

        y = tensor.ivector()
        f = pfunc([y], (y * x), givens={x: 6})
        assert numpy.all(f([1, 1, 1]) == [6, 6, 6])
        assert x.get_value() == 0

        z = tensor.ivector()
        c = z * y
        f = pfunc([y], (c + 7),
                  givens={z: theano._asarray([4, 4, 4], dtype='int32')})
        assert numpy.all(f([1, 1, 1]) == [11, 11, 11])
        assert x.get_value() == 0

    def test_clone0(self):
        x = shared(numpy.asarray([4, 4, 4]))
        y = shared(numpy.asarray([4, 4, 4]))
        z = shared(numpy.asarray([2, 2, 2]))
        up = pfunc([], [], updates={
            x: (x * 5),
            y: ((x * 5) + y),
            z: (((x * 5) + y) ** z)})

        up()
        assert numpy.all(x.get_value() == 20)
        assert numpy.all(y.get_value() == 24)
        assert numpy.all(z.get_value() == (24 ** 2))

    def test_default_updates(self):
        x = shared(0)
        x.default_update = x + 1

        f = pfunc([], [x])
        f()
        assert x.get_value() == 1

        del x.default_update
        f()
        assert x.get_value() == 2

        g = pfunc([], [x])
        g()
        assert x.get_value() == 2

    def test_no_default_updates(self):
        x = shared(0)
        y = shared(1)
        x.default_update = x + 2

        f1 = pfunc([], [x], no_default_updates=True)
        f1()
        assert x.get_value() == 0

        f2 = pfunc([], [x], no_default_updates=[x])
        f2()
        assert x.get_value() == 0

        f3 = pfunc([], [x], no_default_updates=[x, y])
        f3()
        assert x.get_value() == 0

        f4 = pfunc([], [x], no_default_updates=[y])
        f4()
        assert x.get_value() == 2

        f5 = pfunc([], [x], no_default_updates=[])
        f5()
        assert x.get_value() == 4

        f5 = pfunc([], [x], no_default_updates=False)
        f5()
        assert x.get_value() == 6

        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates=(x))
        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates=x)
        self.assertRaises(TypeError, pfunc, [], [x],
                          no_default_updates='canard')

        g1 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=True)
        g1()
        assert x.get_value() == 5

        g2 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[x])
        g2()
        assert x.get_value() == 4

        g3 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[x, y])
        g3()
        assert x.get_value() == 3

        g4 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[y])
        g4()
        assert x.get_value() == 2

        g5 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[])
        g5()
        assert x.get_value() == 1

        g5 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=False)
        g5()
        assert x.get_value() == 0

    def test_default_updates_expressions(self):
        x = shared(0)
        y = shared(1)
        a = lscalar('a')

        z = a * x
        x.default_update = x + y

        f1 = pfunc([a], z)
        f1(12)
        assert x.get_value() == 1

        f2 = pfunc([a], z, no_default_updates=True)
        assert f2(7) == 7
        assert x.get_value() == 1

        f3 = pfunc([a], z, no_default_updates=[x])
        assert f3(9) == 9
        assert x.get_value() == 1

    def test_default_updates_multiple(self):
        x = shared(0)
        y = shared(1)

        x.default_update = x - 1
        y.default_update = y + 1

        f1 = pfunc([], [x, y])
        f1()
        assert x.get_value() == -1
        assert y.get_value() == 2

        f2 = pfunc([], [x, y], updates=[(x, (x - 2))], no_default_updates=[y])
        f2()
        assert x.get_value() == -3
        assert y.get_value() == 2

        f3 = pfunc([], [x, y], updates=[(x, (x - 2))], no_default_updates=True)
        f3()
        assert x.get_value() == -5
        assert y.get_value() == 2

        f4 = pfunc([], [x, y], updates=[(y, (y - 2))])
        f4()
        assert x.get_value() == -6
        assert y.get_value() == 0

    def test_default_updates_chained(self):
        x = shared(2)
        y = shared(1)
        z = shared(-1)

        x.default_update = x - y
        y.default_update = z
        z.default_update = z - 1

        f1 = pfunc([], [x])
        f1()
        assert x.get_value() == 1
        assert y.get_value() == -1
        assert z.get_value() == -2

        f2 = pfunc([], [x, y])
        f2()
        assert x.get_value() == 2
        assert y.get_value() == -2
        assert z.get_value() == -3

        f3 = pfunc([], [y])
        f3()
        assert x.get_value() == 2
        assert y.get_value() == -3
        assert z.get_value() == -4

        f4 = pfunc([], [x, y], no_default_updates=[x])
        f4()
        assert x.get_value() == 2
        assert y.get_value() == -4
        assert z.get_value() == -5

        f5 = pfunc([], [x, y, z], no_default_updates=[z])
        f5()
        assert x.get_value() == 6
        assert y.get_value() == -5
        assert z.get_value() == -5

    def test_default_updates_input(self):
        x = shared(0)
        y = shared(1)
        if theano.configdefaults.python_int_bitwidth() == 32:
            a = iscalar('a')
        else:
            a = lscalar('a')

        x.default_update = y
        y.default_update = y + a

        f1 = pfunc([], x, no_default_updates=True)
        f1()
        assert x.get_value() == 0
        assert y.get_value() == 1

        f2 = pfunc([], x, no_default_updates=[x])
        f2()
        assert x.get_value() == 0
        assert y.get_value() == 1

        f3 = pfunc([], x, no_default_updates=[y])
        f3()
        assert x.get_value() == 1
        assert y.get_value() == 1

        f4 = pfunc([a], x)
        f4(2)
        assert x.get_value() == 1
        assert y.get_value() == 3

        f5 = pfunc([], x, updates={y: (y - 1)})
        f5()
        assert x.get_value() == 3
        assert y.get_value() == 2

        self.assertRaises(theano.gof.MissingInputError, pfunc, [], x)

    def test_default_updates_partial_graph(self):
        a = shared(0)
        a.default_update = a + 1  # Increment a each time it is used
        b = 2 * a
        f = pfunc([b], b)
        assert a.get_value() == 0
        f(21)
        assert a.get_value() == 0

    def test_givens_replaces_shared_variable(self):
        a = shared(1., 'a')
        a.default_update = a + 3.
        b = tensor.dscalar('b')
        c = a + 10
        f = pfunc([b], c, givens={a: b})

        assert len(f.maker.fgraph.inputs) == 1
        assert len(f.maker.fgraph.outputs) == 1

    def test_givens_replaces_shared_variable2(self):
        a = shared(1., 'a')
        a.default_update = a + 3
        c = a + 10
        f = pfunc([], c, givens={a: (a + 10)})

        assert f() == 21
        assert f() == 34

    def test_duplicate_inputs(self):
        x = theano.tensor.lscalar('x')
        self.assertRaises(theano.compile.UnusedInputError,
                          theano.function, [x, x, x], x)

    def test_update_same(self):
        a = shared(1., 'a')
        b = shared(numpy.ones((2, 3)), 'b')

        f = theano.function([], [], updates=[(a, a), (b, (2 * b))])
        g = theano.function([], [], updates=[(a, (a * 2)), (b, b)])

        f()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()
        g()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()

    def test_update_equiv(self):
        a = shared(1., 'a')
        b = shared(numpy.ones((2, 3)), 'b')

        f = theano.function([], [], updates=[(a, a), (b, (2 * b - b))])
        g = theano.function([], [], updates=[(a, (a * 2 - a)), (b, b)])

        f()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()
        g()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()


class Test_aliasing_rules(unittest.TestCase):
    """
    1. Theano manages its own memory space, which typically does not overlap
    with the memory of normal python variables that the user uses.

    2. shared variables are allocated in this memory space, as are the
    temporaries used for Function evalution.

    3. Physically, this managed memory space may be spread across the host,
    on a GPU device(s), or even on a remote machine.

    4. Theano assumes that shared variables are never aliased to one another,
    and tries to make it impossible to accidentally alias them.

    5. Theano's managed data is constant while Theano Functions are not running
    and theano library code is not running.

    6. The default behaviour of Function is to return user-space values for
    outputs, but this can be overridden (borrow=True) for better performance,
    in which case the returned value may be aliased to managed memory, and
    potentially invalidated by the next Theano Function call or call to theano
    library code.
    """

    def shared(self, x):
        return tensor._shared(x)

    def test_shared_constructor_copies(self):
        orig_a = numpy.zeros((2, 2))
        A = self.shared(orig_a)
        assert not numpy.may_share_memory(orig_a, data_of(A))

        assert not numpy.may_share_memory(A.get_value(borrow=False),
                                          data_of(A))

    def test_sparse_input_aliasing_affecting_inplace_operations(self):
        try:
            import scipy.sparse as sp
        except ImportError:
            pass

        from theano.sparse import enable_sparse
        if not enable_sparse:
            raise SkipTest('Optional package sparse disabled')

        from theano import sparse


        x = sparse.SparseType('csc', dtype='float64')()
        y = sparse.SparseType('csc', dtype='float64')()
        f = theano.function([theano.In(x, mutable=True),
                             theano.In(y, mutable=True)],
                            (x + y) + (x + y))

        m = sp.csc_matrix(numpy.asarray(
            [[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]], dtype='float64'))
        bogus_vals = f(m, m)

        m = sp.csc_matrix(numpy.asarray(
            [[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]], dtype='float64'))
        m_copy = m.copy()
        vals = f(m, m_copy)

        assert numpy.allclose(vals.todense(), bogus_vals.todense())

    def test_input_aliasing_affecting_inplace_operations(self):

        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        m1 = theano.tensor.dmatrix()
        m2 = theano.tensor.dmatrix()
        f = theano.function([theano.In(x, mutable=True),
                             theano.In(y, mutable=True),
                             theano.In(m1, mutable=True),
                             theano.In(m2, mutable=True)],
                            theano.dot((x * 2), m1) + theano.dot((y * 3), m2))

        v = numpy.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = numpy.asarray([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]], dtype='float64')
        bogus_vals = f(v, v, m, m)

        v = numpy.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = numpy.asarray([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]], dtype='float64')
        m_copy = m.copy()
        v_copy = v.copy()
        vals = f(v, v_copy, m, m_copy)

        assert numpy.allclose(vals, bogus_vals)

    def test_partial_input_aliasing_affecting_inplace_operations(self):

        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        z = theano.tensor.dvector()
        m1 = theano.tensor.dmatrix()
        m2 = theano.tensor.dmatrix()
        m3 = theano.tensor.dmatrix()


        f = theano.function(
            [theano.In(x, mutable=True),
             theano.In(y, mutable=True),
             theano.In(z, mutable=True),
             theano.In(m1, mutable=True),
             theano.In(m2, mutable=True),
             theano.In(m3, mutable=True)],
            (theano.dot((x * 2), m1) + theano.dot((y * 3), m2) +
             theano.dot((z * 4), m3)))

        v = numpy.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = numpy.asarray([[1, 0],
                           [0, 1]], dtype='float64')
        bogus_vals = f(v[:2], v[1:3], v[2:4], m, m, m)

        v = numpy.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = numpy.asarray([[1, 0],
                           [0, 1]], dtype='float64')
        m_copy1 = m.copy()
        v_copy1 = v.copy()
        m_copy2 = m.copy()
        v_copy2 = v.copy()
        vals = f(v[:2], v_copy1[1:3], v_copy2[2:4], m, m_copy1, m_copy2)

        assert numpy.allclose(vals, bogus_vals)

    def test_potential_output_aliasing_induced_by_updates(self):

        A = self.shared(numpy.zeros((2, 2)))
        B = self.shared(numpy.zeros((2, 2)))
        C = numpy.zeros((2, 2))
        D = tensor.dmatrix()
        DD = D + 5

        f = pfunc([D], [], updates=[(A, D), (B, D)])
        f(C)

        assert not numpy.may_share_memory(data_of(A), data_of(B))
        f = pfunc([D], [], updates=[(A, D[:]), (B, D)])
        f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))
        f = pfunc([D], [], updates=[(A, (D + 5)), (B, D[:])])
        f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))

        f = pfunc([D], [], updates=[(A, (D + 5)), (B, D)])
        f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))

        f = pfunc([D], DD, updates=[(A, DD[:1]), (B, DD)])
        R = f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))
        assert not numpy.may_share_memory(R, data_of(B))
        assert not numpy.may_share_memory(R, data_of(A))

        f = pfunc([D], DD, updates=[(A, DD[:1]), (B, (DD[:1] * 2))])
        R = f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))
        assert not numpy.may_share_memory(R, data_of(B))
        assert not numpy.may_share_memory(R, data_of(A))

        f = pfunc([D], (DD * 4),
                  updates=[(A, (DD[:1] * 3)), (B, (DD[:1] * 2))])
        R = f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))
        assert not numpy.may_share_memory(R, data_of(B))
        assert not numpy.may_share_memory(R, data_of(A))

        f = pfunc([D], (DD * 4),
                  updates=[(A, (DD[:1] * 3)), (B, (DD[:1] * 3))])
        R = f(C)
        assert not numpy.may_share_memory(data_of(A), data_of(B))
        assert not numpy.may_share_memory(R, data_of(B))
        assert not numpy.may_share_memory(R, data_of(A))

    def test_no_aliasing_0(self):
        A = self.shared(numpy.zeros((2, 2)) + .5)
        B = self.shared(numpy.zeros((2, 2)) - .5)
        f = pfunc([], [], updates=[(A, B)])
        f()
        assert not numpy.may_share_memory(data_of(A), data_of(B))

    def test_no_aliasing_1(self):
        A = self.shared(numpy.zeros((2, 2)) + .5)
        B = self.shared(numpy.zeros((2, 2)) - .5)
        C = tensor.dmatrix()
        f = pfunc([C], [], updates=[(A, B), (B, C)])
        z = numpy.zeros((2, 2))
        f(z)
        assert not numpy.may_share_memory(data_of(A), data_of(B))
        assert not numpy.may_share_memory(z, data_of(B))
        assert numpy.all(data_of(B) == z)

    def test_no_aliasing_2(self):
        orig_a = numpy.zeros((2, 2)) + .5
        orig_b = numpy.zeros((2, 2)) - .5
        A = self.shared(orig_a)
        B = self.shared(orig_b)

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = pfunc([], [], updates=[(A, B), (B, A)])
        f()
        assert numpy.all(data_of(A) == -.5)
        assert numpy.all(data_of(B) == +.5)

        assert not numpy.may_share_memory(data_of(A), data_of(B))

        assert numpy.may_share_memory(data_of(A), data_of_b)
        assert numpy.may_share_memory(data_of(B), data_of_a)

    def test_no_aliasing_2b(self):

        orig_a = numpy.zeros((2, 2)) + .5
        orig_b = numpy.zeros((2, 2)) - .5
        A = self.shared(orig_a)
        B = self.shared(orig_b)

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = pfunc([], [], updates=[(A, B[:, ::-1]), (B, A.T)])
        f()
        assert numpy.all(data_of(A) == -.5)
        assert numpy.all(data_of(B) == +.5)

        assert not numpy.may_share_memory(data_of(A), data_of(B))

        if theano.config.mode not in [
                'DebugMode', 'DEBUG_MODE', 'FAST_COMPILE']:
            assert numpy.all(data_of(A) < 5)
            data_of_b += 10
            assert numpy.all(data_of(A) > 5)
            data_of_b -= 10

            assert numpy.all(data_of(B) < 5)
            data_of_a += 10
            assert numpy.all(data_of(B) > 5)
            data_of_a -= 10

            assert numpy.may_share_memory(data_of(A), data_of_b)
            assert numpy.may_share_memory(data_of(B), data_of_a)



class Test_rebuild_strict(unittest.TestCase):
    def test1(self):
        w = tensor.imatrix()
        x, y = tensor.ivectors('x', 'y')
        z = x * y
        f = theano.function([w, y], z, givens=[(x, w)], rebuild_strict=False)
        z_val = f(numpy.ones((3, 5), dtype='int32'), numpy.arange(5, dtype='int32'))
        assert z_val.ndim == 2
        assert numpy.all(z_val == numpy.ones((3, 5)) * numpy.arange(5))


if __name__ == '__main__':
    theano.config.mode = 'FAST_COMPILE'
    Test_pfunc().test_default_scalar_container()
