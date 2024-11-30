from __future__ import absolute_import, print_function, division
import unittest
import theano
from theano import tensor, function, Variable, Generic
import numpy
import os


class T_load_tensor(unittest.TestCase):
    def setUp(self):
        self.data = numpy.arange(5, dtype=numpy.int32)
        self.filename = os.path.join(
            theano.config.compiledir,
            "_test.npy")
        numpy.save(self.filename, self.data)

    def test0(self):
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,))
        y = x * 2
        fn = function([path], y)
        assert (fn(self.filename) == (self.data * 2)).all()

    def test_invalid_modes(self):
        path = Variable(Generic())
        for mmap_mode in ('r+', 'r', 'w+', 'toto'):
            self.assertRaises(ValueError,
                    tensor.load, path, 'int32', (False,), mmap_mode)

    def test1(self):
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,), 'c')
        y = (x ** 2).sum()
        fn = function([path], y)
        assert (fn(self.filename) == (self.data ** 2).sum()).all()
        assert (fn(self.filename) == (self.data ** 2).sum()).all()

    def test_memmap(self):
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,), mmap_mode='c')
        fn = function([path], x)
        assert type(fn(self.filename)) == numpy.core.memmap

    def tearDown(self):
        os.remove(os.path.join(
            theano.config.compiledir,
            "_test.npy"))
