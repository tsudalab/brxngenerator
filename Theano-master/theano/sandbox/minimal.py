from __future__ import absolute_import, print_function, division

import unittest

import numpy

from theano import gof, tensor, function
from theano.tests import unittest_tools as utt


class Minimal(gof.Op):


    def __init__(self):

        super(Minimal, self).__init__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, *args):
        return gof.Apply(op=self, inputs=args, outputs=[tensor.lscalar()])

    def perform(self, node, inputs, out_):
        output, = out_

        print("perform got %i arguments" % len(inputs))

        print("Max of input[0] is ", numpy.max(inputs[0]))

        output[0] = numpy.asarray(0, dtype='int64')

minimal = Minimal()




class T_minimal(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed(666))

    def test0(self):
        A = tensor.matrix()
        b = tensor.vector()

        print('building function')
        f = function([A, b], minimal(A, A, b, b, A))
        print('built')

        Aval = self.rng.randn(5, 5)
        bval = numpy.arange(5, dtype=float)
        f(Aval, bval)
        print('done')
