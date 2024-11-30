from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest
import numpy
import theano.sparse
if not theano.sparse.enable_sparse:
    raise SkipTest('Optional package sparse disabled')
from theano.sparse.utils import hash_from_sparse
from theano.sparse.tests.test_basic import as_sparse_format



def test_hash_from_sparse():
    hashs = []
    rng = numpy.random.rand(5, 5)

    for format in ['csc', 'csr']:
        rng = as_sparse_format(rng, format)
        for data in [[[-2]], [[-1]], [[0]], [[1]], [[2]],
                     numpy.zeros((1, 5)), numpy.zeros((1, 6)),
                     numpy.arange(25).reshape(5, 5),
                     numpy.arange(25).reshape(5, 5).T,
                     numpy.zeros((5, 5), dtype="uint32"),
                     numpy.zeros((5, 5), dtype="int32"),
                     rng, rng[1:], rng[:4], rng[1:3],
                     ]:
            data = as_sparse_format(data, format)

            hashs.append(hash_from_sparse(data))

        assert hash_from_sparse(rng[1:]) == hash_from_sparse(rng[1:].copy())
        assert hash_from_sparse(rng[1:3]) == hash_from_sparse(rng[1:3].copy())
        assert hash_from_sparse(rng[:4]) == hash_from_sparse(rng[:4].copy())

    assert len(set(hashs)) == len(hashs)
