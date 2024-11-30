from __future__ import absolute_import, print_function, division
import itertools
import unittest
import numpy
import theano
from theano import tensor
from theano.scan_module.scan_utils import equal_computations, map_variables
from theano.tensor.type_other import NoneConst


def test_equal_compuations():
    c = NoneConst
    assert equal_computations([c], [c])
    m = theano.tensor.matrix()
    max_argmax1 = theano.tensor.max_and_argmax(m)
    max_argmax2 = theano.tensor.max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)



class TestMapVariables(unittest.TestCase):
    @staticmethod
    def replacer(graph):
        return getattr(graph.tag, "replacement", graph)

    def test_leaf(self):
        a = tensor.scalar("a")
        b = tensor.scalar("b")
        c = tensor.scalar("c")

        b.tag.replacement = c

        u = a + b
        v, = map_variables(self.replacer, [u])

        assert u.owner.inputs == [a, b]
        assert v.owner.inputs == [a, c]

    def test_leaf_inside_scan(self):
        x = tensor.vector('x')
        y = tensor.scalar('y')
        z = tensor.scalar('z')

        y.tag.replacement = z

        s, _ = theano.scan(lambda x: x * y, sequences=x)
        s2, = map_variables(self.replacer, [s])

        f = theano.function([x, y, z], [s, s2])
        rval = f(x=numpy.array([1, 2, 3], dtype=numpy.float32), y=1, z=2)
        assert numpy.array_equal(rval, [[1, 2, 3], [2, 4, 6]])

    def test_scan(self):
        x = tensor.vector('x')

        outer = tensor.scalar("outer")
        shared = theano.shared(
            numpy.array(1., dtype=theano.config.floatX),
            name="shared")
        constant = tensor.constant(1, name="constant")

        z = outer * (shared + constant)

        def step(x, a):
            r = a + x
            r.tag.replacement = z * (a - x)
            return r

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[numpy.array(0.)])
        t = z * s
        s2, = map_variables(self.replacer, [t])
        t2 = z * s2

        f = theano.function([x, outer], [t, t2])
        rval = f(x=numpy.array([1, 2, 3], dtype=numpy.float32), outer=0.5)
        assert numpy.array_equal(rval, [[1, 3, 6], [-1, -3, -6]])

    def test_scan_with_shared_update(self):
        x = tensor.vector('x')

        counter = theano.shared(0, name="shared")
        counter.update = counter + 1

        def step(x, a):
            r = a + x
            r.tag.replacement = counter * (a - x)
            return r

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[numpy.array(0.)])
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [s])

    def test_scan_with_shared_update2(self):
        x = tensor.vector('x')

        counter = theano.shared(0, name="shared")
        counter.update = counter + 1

        def step(x, a):
            r = a + x
            r.tag.replacement = counter * (a - x)
            return r + counter

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[numpy.array(0.)])
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [s])

    def test_opfromgraph(self):
        outer = tensor.scalar("outer")
        shared = theano.shared(
            numpy.array(1., dtype=theano.config.floatX),
            name="shared")
        constant = tensor.constant(1., name="constant")
        z = outer * (shared + constant)

        a = tensor.scalar()
        b = tensor.scalar()
        r = a + b
        r.tag.replacement = z * (a - b)

        c = tensor.scalar()
        d = tensor.scalar()
        u = theano.OpFromGraph([a, b], [r])(c, d)
        t = z * u
        v, = map_variables(self.replacer, [t])
        t2 = z * v

        f = theano.function([c, d, outer], [t, t2])
        for m, n in itertools.combinations(range(10), 2):
            assert f(m, n, outer=0.5) == [m + n, m - n]

        shared.update = shared + 1
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [t])
