from __future__ import absolute_import, print_function, division
import os
import sys
import traceback
import warnings

import numpy
from nose.plugins.skip import SkipTest
import unittest

import theano
from theano import config
from theano import scalar
from theano import tensor as T
from theano.gof import Apply, Op
from theano.gof import utils
from theano.tensor.basic import _allclose


class IncOneC(Op):
    """An Op with only a C (c_code) implementation"""
    __props__ = ()

    def make_node(self, input):
        input = scalar.as_scalar(input)
        output = input.type()
        return Apply(self, [input], [output])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        z, = outputs
        return "%(z)s = %(x)s + 1;" % locals()


class TestComputeTestValue(unittest.TestCase):

    def test_variable_only(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3, 4).astype(config.floatX)
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4, 5).astype(config.floatX)

            z = T.dot(x, y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([x, y], z)
            assert _allclose(f(x.tag.test_value, y.tag.test_value),
                             z.tag.test_value)

            y.tag.test_value = numpy.random.rand(6, 5).astype(config.floatX)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_compute_flag(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            x = T.matrix('x')
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4, 5).astype(config.floatX)

            theano.config.compute_test_value = 'off'
            z = T.dot(x, y)
            assert not hasattr(z.tag, 'test_value')

            theano.config.compute_test_value = 'raise'
            self.assertRaises(ValueError, T.dot, x, y)

            theano.config.compute_test_value = 'warn'
            warnings.simplefilter('error', UserWarning)
            try:
                self.assertRaises(UserWarning, T.dot, x, y)
            finally:
                warnings.simplefilter('default', UserWarning)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_string_var(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3, 4).astype(config.floatX)
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4, 5).astype(config.floatX)

            z = theano.shared(numpy.random.rand(5, 6).astype(config.floatX))

            out = T.dot(T.dot(x, y), z)
            assert hasattr(out.tag, 'test_value')
            tf = theano.function([x, y], out)
            assert _allclose(
                tf(x.tag.test_value, y.tag.test_value),
                out.tag.test_value)

            def f(x, y, z):
                return T.dot(T.dot(x, y), z)

            z.set_value(numpy.random.rand(7, 6).astype(config.floatX))
            self.assertRaises(ValueError, f, x, y, z)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_shared(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3, 4).astype(config.floatX)
            y = theano.shared(numpy.random.rand(4, 6).astype(config.floatX),
                              'y')

            z = T.dot(x, y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([x], z)
            assert _allclose(f(x.tag.test_value), z.tag.test_value)

            y.set_value(numpy.random.rand(5, 6).astype(config.floatX))
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_ndarray(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = numpy.random.rand(2, 3).astype(config.floatX)
            y = theano.shared(numpy.random.rand(3, 6).astype(config.floatX),
                              'y')

            z = T.dot(x, y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([], z)
            assert _allclose(f(), z.tag.test_value)

            x = numpy.random.rand(2, 4).astype(config.floatX)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_empty_elemwise(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = theano.shared(numpy.random.rand(0, 6).astype(config.floatX),
                              'x')

            z = (x + 2) * 3
            assert hasattr(z.tag, 'test_value')
            f = theano.function([], z)
            assert _allclose(f(), z.tag.test_value)

        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_constant(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.constant(numpy.random.rand(2, 3), dtype=config.floatX)
            y = theano.shared(numpy.random.rand(3, 6).astype(config.floatX),
                              'y')

            z = T.dot(x, y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([], z)
            assert _allclose(f(), z.tag.test_value)

            x = T.constant(numpy.random.rand(2, 4), dtype=config.floatX)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_incorrect_type(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.fmatrix('x')
            x.tag.test_value = numpy.random.rand(3, 4)
            y = T.dmatrix('y')
            y.tag.test_value = numpy.random.rand(4, 5)

            self.assertRaises(TypeError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_overided_function(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            config.compute_test_value = "raise"
            x = T.matrix()
            x.tag.test_value = numpy.zeros((2, 3), dtype=config.floatX)
            y = T.matrix()
            y.tag.test_value = numpy.zeros((2, 2), dtype=config.floatX)
            self.assertRaises(ValueError, x.__mul__, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_scan(self):
        """
        Test the compute_test_value mechanism Scan.
        """
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'
            k = T.iscalar("k")
            A = T.vector("A")
            k.tag.test_value = 3
            A.tag.test_value = numpy.random.rand(5).astype(config.floatX)

            def fx(prior_result, A):
                return prior_result * A
            result, updates = theano.scan(fn=fx,
                                          outputs_info=T.ones_like(A),
                                          non_sequences=A,
                                          n_steps=k)

            final_result = result[-1]
            assert hasattr(final_result.tag, 'test_value')
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_scan_err1(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            k = T.iscalar("k")
            A = T.matrix("A")
            k.tag.test_value = 3
            A.tag.test_value = numpy.random.rand(5, 3).astype(config.floatX)

            def fx(prior_result, A):
                return T.dot(prior_result, A)

            try:
                theano.scan(
                    fn=fx,
                    outputs_info=T.ones_like(A),
                    non_sequences=A,
                    n_steps=k)
                assert False
            except ValueError:
                tb = sys.exc_info()[2]
                frame_info = traceback.extract_tb(tb)[-5]
                expected = 'test_compute_test_value.py'
                assert os.path.split(frame_info[0])[1] == expected, frame_info
                assert frame_info[2] == 'fx'

        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_scan_err2(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            k = T.iscalar("k")
            A = T.matrix("A")
            k.tag.test_value = 3
            A.tag.test_value = numpy.random.rand(5, 3).astype(config.floatX)

            def fx(prior_result, A):
                return T.dot(prior_result, A)

            self.assertRaises(ValueError,
                              theano.scan,
                              fn=fx,
                              outputs_info=T.ones_like(A.T),
                              non_sequences=A,
                              n_steps=k)

            try:
                theano.scan(
                    fn=fx,
                    outputs_info=T.ones_like(A.T),
                    non_sequences=A,
                    n_steps=k)
                assert False
            except ValueError as e:
                assert (str(e).startswith("shape mismatch") or
                        str(e).startswith("operands could not be broadcast "
                                          "together with shapes") or
                        str(e).startswith("could not broadcast input"))

        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_no_c_code(self):
        class IncOnePython(Op):
            """An Op with only a Python (perform) implementation"""
            __props__ = ()

            def make_node(self, input):
                input = scalar.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def perform(self, node, inputs, outputs):
                input, = inputs
                output, = outputs
                output[0] = input + 1

        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            i = scalar.int32('i')
            i.tag.test_value = 3

            o = IncOnePython()(i)

            self.assertRaises(
                (NotImplementedError, utils.MethodNotDefined),
                o.owner.op.c_code,
                o.owner, 'o', ['x'], 'z', {'fail': ''})

            assert hasattr(o.tag, 'test_value')
            assert o.tag.test_value == 4

        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_no_perform(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")

        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            i = scalar.int32('i')
            i.tag.test_value = 3

            o = IncOneC()(i)

            self.assertRaises((NotImplementedError, utils.MethodNotDefined),
                              o.owner.op.perform,
                              o.owner, 0, [None])

            assert hasattr(o.tag, 'test_value')
            assert o.tag.test_value == 4

        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_disabled_during_compilation(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'
            init_Mu1 = theano.shared(
                numpy.zeros((5,), dtype=config.floatX)).dimshuffle('x', 0)

            theano.function([], outputs=[init_Mu1])
        finally:
            theano.config.compute_test_value = orig_compute_test_value
