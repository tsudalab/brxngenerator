from __future__ import absolute_import, print_function, division
import traceback

import numpy
from six import integer_types

import theano.tensor.basic
from theano.tensor.basic import TensorType, _tensor_py_operators
from theano.compile import shared_constructor, SharedVariable


def load_shared_variable(val):
    """
    This function is only here to keep some pickles loading
    after a failed fix done in August 2011.
    It can be removed after sufficient time has passed.

    """
    return tensor_constructor(val)


class TensorSharedVariable(_tensor_py_operators, SharedVariable):
    pass


@shared_constructor
def tensor_constructor(value, name=None, strict=False, allow_downcast=None,
                       borrow=False, broadcastable=None, target='cpu'):
    """
    SharedVariable Constructor for TensorType.

    Notes
    -----
    Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
    The optional `broadcastable` argument will override this default.

    """
    if target != 'cpu':
        raise TypeError('not for cpu')

    if not isinstance(value, numpy.ndarray):
        raise TypeError()

    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = TensorType(value.dtype, broadcastable=broadcastable)
    return TensorSharedVariable(type=type,
                                value=numpy.array(value, copy=(not borrow)),
                                name=name,
                                strict=strict,
                                allow_downcast=allow_downcast)


class ScalarSharedVariable(_tensor_py_operators, SharedVariable):
    pass


@shared_constructor
def scalar_constructor(value, name=None, strict=False, allow_downcast=None,
                       borrow=False, target='cpu'):
    """
    SharedVariable constructor for scalar values. Default: int64 or float64.

    Notes
    -----
    We implement this using 0-d tensors for now.

    We ignore the borrow parameter as we convert ``value`` to an
    ndarray (this is a new object). This respects the semantic of
    borrow, as it is a hint to Theano that we can reuse it.

    """
    if target != 'cpu':
        raise TypeError('not for cpu')

    if not isinstance(value, (numpy.number, float, integer_types, complex)):
        raise TypeError()
    try:
        dtype = value.dtype
    except Exception:
        dtype = numpy.asarray(value).dtype

    dtype = str(dtype)
    value = theano._asarray(value, dtype=dtype)
    tensor_type = TensorType(dtype=str(value.dtype), broadcastable=[])

    try:
        rval = ScalarSharedVariable(type=tensor_type,
                                    value=numpy.array(value, copy=True),
                                    name=name,
                                    strict=strict,
                                    allow_downcast=allow_downcast)
        return rval
    except Exception:
        traceback.print_exc()
        raise
