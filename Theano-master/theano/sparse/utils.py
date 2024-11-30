from __future__ import absolute_import, print_function, division
from theano.gof.utils import hash_from_code


def hash_from_sparse(data):

    return hash_from_code(hash_from_code(data.data) +
                          hash_from_code(data.indices) +
                          hash_from_code(data.indptr) +
                          hash_from_code(str(data.shape)) +
                          hash_from_code(str(data.dtype)) +
                          hash_from_code(data.format))
