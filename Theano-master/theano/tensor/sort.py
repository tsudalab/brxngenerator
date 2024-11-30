from __future__ import absolute_import, print_function, division
import numpy as np
import theano
from theano.tensor.basic import mul, arange


class SortOp(theano.Op):
    """
    This class is a wrapper for numpy sort function.

    """

    __props__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __str__(self):
        return self.__class__.__name__ + "{%s, %s}" % (self.kind,
                                                       str(self.order))

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        axis = theano.tensor.as_tensor_variable(axis)
        out_type = input.type()
        return theano.Apply(self, [input, axis], [out_type])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = np.sort(a, axis, self.kind, self.order)

    def infer_shape(self, node, inputs_shapes):
        if (isinstance(node.inputs[1], theano.Constant) and
                node.inputs[1].data is None):
            return [(mul(*inputs_shapes[0]),)]
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        a, axis = inputs
        indices = self.__get_argsort_indices(a, axis)
        inp_grad = output_grads[0][tuple(indices)]
        axis_grad = theano.gradient.grad_undefined(
            self, 1, axis,
            "The gradient of sort is not defined "
            "with respect to the integer axes itself")
        return [inp_grad, axis_grad]

    def __get_expanded_dim(self, a, axis, i):
        index_shape = [1] * a.ndim
        index_shape[i] = a.shape[i]
        index_val = arange(a.shape[i]).reshape(index_shape)
        return index_val

    def __get_argsort_indices(self, a, axis):
        """
        Calculates indices which can be used to reverse sorting operation of
        "a" tensor along "axis".

        Returns
        -------
        1d array if axis is None
        list of lenght len(a.shape) otherwise

        """

        idx = argsort(a, axis, kind=self.kind, order=self.order)
        rev_idx = argsort(idx, axis, kind=self.kind, order=self.order)
        indices = []
        axis_data = theano.tensor.switch(theano.tensor.ge(axis.data, 0),
                                         axis.data, a.ndim + axis.data)
        for i in range(a.ndim):
            index_val = theano.tensor.switch(theano.tensor.eq(i, axis_data),
                                             rev_idx,
                                             self.__get_expanded_dim(a,
                                                                     axis, i))
            indices.append(index_val)
        return indices
    """
    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    """


def sort(a, axis=-1, kind='quicksort', order=None):
    """

    Parameters
    ----------
    a : Tensor
        Tensor to be sorted
    axis : Tensor
        Axis along which to sort. If None, the array is flattened before
        sorting.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.
    order : list, optional
        When `a` is a structured array, this argument specifies which
        fields to compare first, second, and so on. This list does not
        need to include all of the fields.

    Returns
    -------
    array
        A sorted copy of an array.

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return SortOp(kind, order)(a, axis)


class ArgSortOp(theano.Op):
    """
    This class is a wrapper for numpy argsort function.

    """

    __props__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __str__(self):
        return (self.__class__.__name__ +
                "{%s, %s}" % (self.kind, str(self.order)))

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        axis = theano.tensor.as_tensor_variable(axis)
        bcast = input.type.broadcastable
        return theano.Apply(self, [input, axis], [theano.tensor.TensorType(
            dtype="int64", broadcastable=bcast)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = theano._asarray(np.argsort(a, axis, self.kind, self.order),
                               dtype=node.outputs[0].dtype)

    def infer_shape(self, node, inputs_shapes):
        if (isinstance(node.inputs[1], theano.Constant) and
                node.inputs[1].data is None):
            return [(mul(*inputs_shapes[0]),)]
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        inp, axis = inputs
        inp_grad = inp.zeros_like()
        axis_grad = theano.gradient.grad_undefined(
            self, 1, axis,
            "argsort is not defined for non-integer axes so"
            " argsort(x, axis+eps) is undefined")
        return [inp_grad, axis_grad]
    """
    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    """


def argsort(a, axis=-1, kind='quicksort', order=None):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm
    specified by the kind keyword.  It returns an array of indices of
    the same shape as a that index data along the given axis in sorted
    order.

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return ArgSortOp(kind, order)(a, axis)
