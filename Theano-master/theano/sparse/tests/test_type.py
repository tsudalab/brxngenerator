from __future__ import absolute_import, print_function, division


def test_sparse_type():
    import theano.sparse
    assert hasattr(theano.sparse, "SparseType")
