from __future__ import absolute_import, print_function, division
import unittest
import numpy
from nose.plugins.skip import SkipTest

import theano
from theano import tensor

from theano.sandbox.cuda.var import float32_shared_constructor as f32sc
from theano.sandbox.cuda import CudaNdarrayType, cuda_available
import theano.sandbox.cuda as cuda
if cuda_available == False:
    raise SkipTest('Optional package cuda disabled')


def test_float32_shared_constructor():

    npy_row = numpy.zeros((1, 10), dtype='float32')

    def eq(a, b):
        return a == b

    assert (f32sc(npy_row).type == CudaNdarrayType((False, False)))

    assert eq(
            f32sc(npy_row, broadcastable=(True, False)).type,
            CudaNdarrayType((True, False)))
    assert eq(
            f32sc(npy_row, broadcastable=[True, False]).type,
            CudaNdarrayType((True, False)))
    assert eq(
            f32sc(npy_row, broadcastable=numpy.array([True, False])).type,
            CudaNdarrayType([True, False]))

    assert eq(
            f32sc(numpy.zeros((2, 3, 4, 5), dtype='float32')).type,
            CudaNdarrayType((False,) * 4))


def test_givens():
    data = numpy.float32([1, 2, 3, 4])
    x = f32sc(data)
    y = x ** 2
    f = theano.function([], y, givens={x: x + 1})
    f()


class T_updates(unittest.TestCase):

    def test_1(self):
        data = numpy.float32([1, 2, 3, 4])
        x = f32sc(data)
        y = x ** 2
        f = theano.function([], y, updates=[(x, x + 1)])
        f()

        f = theano.function([], y, updates=[(x, cuda.gpu_from_host(x + 1))])
        f()

    def test_2(self):
        data = numpy.random.rand(10, 10).astype('float32')
        output_var = f32sc(name="output",
                value=numpy.zeros((10, 10), 'float32'))

        x = tensor.fmatrix('x')
        output_updates = [(output_var, x ** 2)]
        output_givens = {x: data}
        output_func = theano.function(inputs=[], outputs=[],
                updates=output_updates, givens=output_givens)
        output_func()

    def test_err_ndim(self):
        data = numpy.random.rand(10, 10).astype('float32')
        output_var = f32sc(name="output", value=data)

        self.assertRaises(TypeError, theano.function, inputs=[], outputs=[],
                          updates=[(output_var,
                                   output_var.sum())])

    def test_err_broadcast(self):
        data = numpy.random.rand(10, 10).astype('float32')
        output_var = f32sc(name="output", value=data)

        self.assertRaises(TypeError, theano.function, inputs=[], outputs=[],
                          updates=[(output_var,
                                   output_var.sum().dimshuffle('x', 'x'))])

    def test_broadcast(self):
        data = numpy.random.rand(10, 10).astype('float32')
        output_var = f32sc(name="output", value=data)

        up = tensor.unbroadcast(output_var.sum().dimshuffle('x', 'x'), 0, 1)
        output_func = theano.function(inputs=[], outputs=[],
                                      updates=[(output_var, up)])
        output_func()

        up = tensor.patternbroadcast(output_var.sum().dimshuffle('x', 'x'),
                                     output_var.type.broadcastable)
        output_func = theano.function(inputs=[], outputs=[],
                                      updates=[(output_var, up)])
        output_func()
