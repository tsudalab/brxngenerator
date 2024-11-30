from __future__ import absolute_import, print_function, division
import string

import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda import cuda_available, GpuOp
from theano.ifelse import ifelse
from theano.misc.pycuda_init import pycuda_available

if cuda_available:
    from theano.sandbox.cuda import (basic_ops, CudaNdarrayType,
                                     CudaNdarray)
if pycuda_available:
    import pycuda.gpuarray

try:
    import scikits.cuda
    from scikits.cuda import fft, cublas
    scikits.cuda.misc.init()
    scikits_cuda_available = True
except (ImportError, Exception):
    scikits_cuda_available = False



class ScikitsCudaOp(GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        raise NotImplementedError

    def make_node(self, inp):
        inp = basic_ops.gpu_contiguous(
            basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2):
        if not scikits_cuda_available:
            raise RuntimeError(
                "scikits.cuda is needed for all GPU fft implementation,"
                " including fftconv.")


class CuFFTOp(ScikitsCudaOp):
    def output_type(self, inp):
        return CudaNdarrayType(
            broadcastable=[False] * (inp.type.ndim + 1))

    def make_thunk(self, node, storage_map, _, _2):
        super(CuFFTOp, self).make_thunk(node, storage_map, _, _2)

        from theano.misc.pycuda_utils import to_gpuarray
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            output_shape = list(input_shape)
            output_shape[-1] = output_shape[-1] // 2 + 1
            output_shape += [2]
            output_shape = tuple(output_shape)

            z = outputs[0]

            if z[0] is None or z[0].shape != output_shape:
                z[0] = CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            output_pycuda = to_gpuarray(z[0])

            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(input_shape[1:], np.float32, np.complex64,
                                   batch=input_shape[0])

            fft.fft(input_pycuda, output_pycuda, plan[0])

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


class CuIFFTOp(ScikitsCudaOp):
    def output_type(self, inp):
        return CudaNdarrayType(
            broadcastable=[False] * (inp.type.ndim - 1))

    def make_thunk(self, node, storage_map, _, _2):
        super(CuIFFTOp, self).make_thunk(node, storage_map, _, _2)

        from theano.misc.pycuda_utils import to_gpuarray
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            output_shape = list(input_shape[:-1])
            output_shape[-1] = (output_shape[-1] - 1) * 2
            output_shape = tuple(output_shape)

            z = outputs[0]

            if z[0] is None or z[0].shape != output_shape:
                z[0] = CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            output_pycuda = to_gpuarray(z[0])

            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(output_shape[1:], np.complex64, np.float32,
                                   batch=output_shape[0])

            fft.ifft(input_pycuda, output_pycuda, plan[0])

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


def to_complex_gpuarray(x, copyif=False):
    """
    Adapted version of theano.misc.pycuda_utils.to_gpuarray that takes
    an array with an extra trailing dimension of length 2 for
    real/imaginary parts, and turns it into a complex64 PyCUDA
    GPUArray.

    """
    if not isinstance(x, CudaNdarray):
        raise ValueError("We can transfer only CudaNdarray "
                         "to pycuda.gpuarray.GPUArray")
    else:
        assert x.shape[-1] == 2

        assert x.dtype == 'float32'

        size = 1
        c_contiguous = True
        for i in range(x.ndim - 1, -1, -1):
            if x.shape[i] == 1:
                continue
            if x._strides[i] != size:
                c_contiguous = False
                break
            size *= x.shape[i]
        if not c_contiguous:
            if copyif:
                x = x.copy()
            else:
                raise ValueError("We were asked to not copy memory, "
                                 "but the memory is not c contiguous.")

        px = pycuda.gpuarray.GPUArray(x.shape[:-1], np.complex64, base=x,
                                      gpudata=x.gpudata)
        return px


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.

    Taken from scikits.cuda tests/test_cublas.py.

    """
    return pycuda.gpuarray.arange(a.ptr, a.ptr + a.shape[0] * a.strides[0],
                                  a.strides[0], dtype=cublas.ctypes.c_void_p)


def sc_complex_dot_batched(bx_gpu, by_gpu, bc_gpu, transa='N', transb='N',
                           handle=None):
    """
    Uses cublasCgemmBatched to compute a bunch of complex dot products
    in parallel.

    """
    if handle is None:
        handle = scikits.cuda.misc._global_cublas_handle

    assert len(bx_gpu.shape) == 3
    assert len(by_gpu.shape) == 3
    assert len(bc_gpu.shape) == 3
    assert bx_gpu.dtype == np.complex64
    assert by_gpu.dtype == np.complex64
    assert bc_gpu.dtype == np.complex64

    bx_shape = bx_gpu.shape
    by_shape = by_gpu.shape

    alpha = np.complex64(1.0)
    beta = np.complex64(0.0)

    transa = string.lower(transa)
    transb = string.lower(transb)

    if transb in ['t', 'c']:
        N, m, k = by_shape
    elif transb in ['n']:
        N, k, m = by_shape
    else:
        raise ValueError('invalid value for transb')

    if transa in ['t', 'c']:
        N2, l, n = bx_shape
    elif transa in ['n']:
        N2, n, l = bx_shape
    else:
        raise ValueError('invalid value for transa')

    if l != k:
        raise ValueError('objects are not aligned')

    if N != N2:
        raise ValueError('batch sizes are not the same')

    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)

    bx_arr = bptrs(bx_gpu)
    by_arr = bptrs(by_gpu)
    bc_arr = bptrs(bc_gpu)

    cublas.cublasCgemmBatched(handle, transb, transa, m, n, k, alpha,
                              by_arr.gpudata, lda, bx_arr.gpudata, ldb,
                              beta, bc_arr.gpudata, ldc, N)


class BatchedComplexDotOp(ScikitsCudaOp):
    """
    This version uses cublasCgemmBatched under the hood, instead of
    doing multiple cublasCgemm calls.

    """

    def make_node(self, inp1, inp2):
        inp1 = basic_ops.gpu_contiguous(
            basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = basic_ops.gpu_contiguous(
            basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 4  # (batch, a, b, real/imag)
        assert inp2.ndim == 4

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        super(BatchedComplexDotOp, self).make_thunk(node, storage_map, _, _2)

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            bx = inputs[0]
            by = inputs[1]

            input_shape_x = bx[0].shape  # (batch, a, b, 2)
            input_shape_y = by[0].shape  # (batch, b, c, 2)

            output_shape = (input_shape_x[0], input_shape_x[1],
                            input_shape_y[2], 2)  # (batch, a, c, 2)

            bz = outputs[0]

            if bz[0] is None or bz[0].shape != output_shape:
                bz[0] = CudaNdarray.zeros(output_shape)

            input_bx_pycuda = to_complex_gpuarray(bx[0])
            input_by_pycuda = to_complex_gpuarray(by[0])
            output_b_pycuda = to_complex_gpuarray(bz[0])

            sc_complex_dot_batched(input_bx_pycuda, input_by_pycuda,
                                   output_b_pycuda)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


cufft = CuFFTOp()
cuifft = CuIFFTOp()
batched_complex_dot = BatchedComplexDotOp()


def mult_and_reduce(input_fft_v, filters_fft_v, input_shape=None,
                    filter_shape=None):
    """

    Parameters
    ----------
    input_fft_v
        It's (b, ic, i0, i1//2 + 1, 2).
    filters_fft_v
        It's (oc, ic, i0, i1//2 + 1, 2).

    """
    if input_shape is None:
        input_shape = input_fft_v.shape  # symbolic

    if filter_shape is None:
        filter_shape = filters_fft_v.shape  # symbolic

    b, ic, i0, i1_f, _ = input_shape
    oc = filter_shape[0]

    input_r = input_fft_v.reshape((b, ic, i0 * i1_f, 2))
    filters_r = filters_fft_v.reshape((oc, ic, i0 * i1_f, 2))

    input_s = input_r.dimshuffle(2, 0, 1, 3)  # (i0 * i1_f, b, ic, 2)
    filters_s = filters_r.dimshuffle(2, 1, 0, 3)  # (i0 * i1_f, ic, oc, 2)

    output_s = batched_complex_dot(input_s, filters_s)

    output_r = output_s.dimshuffle(1, 2, 0, 3)

    output = output_r.reshape((b, oc, i0, i1_f, 2))

    return output


def conv2d_fft(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', pad_last_dim=False):
    """
    Perform a convolution through fft.

    Only support input which will be even on the last dimension
    (width).  All other dimensions can be anything and the filters can
    have an even or odd width.

    If you must use input which has an odd width, you can either pad
    it or use the `pad_last_dim` argument which will do it for you and
    take care to strip the padding before returning.  Don't use this
    argument if you are not sure the input is odd since the padding is
    unconditional and will make even input odd, thus leading to
    problems.

    On valid mode the filters must be smaller than the input.

    Parameters
    ----------
    input
        (b, ic, i0, i1).
    filters
        (oc, ic, f0, f1).
    border_mode : {'valid', 'full'}
    pad_last_dim
        Unconditionally pad the last dimension of the input
        to to turn it from odd to even.  Will strip the
        padding before returning the result.

    """
    if image_shape is None:
        image_shape = input.shape

    if filter_shape is None:
        filter_shape = filters.shape

    b, ic, i0, i1 = image_shape
    oc, ic_, f0, f1 = filter_shape

    if border_mode == 'valid':
        o0 = i0
        if pad_last_dim:
            o1 = i1 + 1
            input_padded = T.zeros((b, ic, o0, o1), dtype='float32')
            input_padded = T.set_subtensor(input_padded[:, :, :i0, :i1],
                                       input)
        else:
            o1 = i1
            input_padded = input

        filters_padded = T.zeros((oc, ic, o0, o1), dtype='float32')
        filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1],
                                         filters)

    elif border_mode == 'full':

        o0 = i0 + 2 * (f0 - 1)
        o1 = i1 + 2 * (f1 - 1)

        if pad_last_dim:
            o1 = o1 + 1


        filters_padded = T.zeros((oc, ic, o0, o1), dtype='float32')
        filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1],
                                         filters)

        input_padded = T.zeros((b, ic, o0, o1), dtype='float32')
        input_padded = T.set_subtensor(input_padded[:, :, (f0 - 1):(f0 - 1 + i0), (f1 - 1):(f1 - 1 + i1)],
                                       input)
    else:
        raise ValueError('invalid mode')

    input_padded = T.opt.Assert("in conv2d_fft: width is not even")(
        input_padded, T.eq(o1 % 2, 0))

    input_flat = input_padded.reshape((b * ic, o0, o1))
    filters_flat = filters_padded.reshape((oc * ic, o0, o1))

    input_fft_flat = cufft(input_flat)  # (b * ic, o0, o1//2 + 1, 2)
    filters_fft_flat = cufft(filters_flat)  # (oc * ic, o0, o1//2 + 1, 2)

    input_fft_v_shape = (b, ic, o0, o1 // 2 + 1, 2)
    filters_fft_v_shape = (oc, ic, o0, o1 // 2 + 1, 2)
    input_fft_v = input_fft_flat.reshape(input_fft_v_shape)
    filters_fft_v = filters_fft_flat.reshape(filters_fft_v_shape)

    output_fft_s = mult_and_reduce(input_fft_v, filters_fft_v,
                                   input_shape=input_fft_v_shape,
                                   filter_shape=filters_fft_v_shape)

    output_fft_flat = output_fft_s.reshape((b * oc, o0, o1 // 2 + 1, 2))

    output_flat = cuifft(output_fft_flat)  # (b * oc, o0, o1)

    output_circ = output_flat.reshape((b, oc, o0, o1))  # circular!

    if border_mode == 'valid':
        output = output_circ[:, :, (f0-1):(f0-1 + i0-f0+1), (f1-1):(f1-1 + i1-f1+1)]
    elif border_mode == 'full':
        output = output_circ[:, :, (f0-1):(f0-1 + i0+f0-1), (f1-1):(f1-1 + i1+f1-1)]
    else:
        raise ValueError('invalid mode')

    output = (1.0 / T.cast(o0 * o1, 'float32')) * output

    return basic_ops.as_cuda_ndarray_variable(output)


def conv3d_fft(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', pad_last_dim=False):
    """
    Perform a convolution through fft.

    Only supports input whose shape is even on the last dimension.
    All other dimensions can be anything and the filters can
    have an even or odd last dimension.

    The semantics associated with the last three dimensions
    are not important as long as they are in the same order between
    the inputs and the filters. For example, when the convolution
    is done on a sequence of images, they could be either
    (duration, height, width) or (height, width, duration).

    If you must use input which has an odd width, you can either pad
    it or use the `pad_last_dim` argument which will do it for you and
    take care to strip the padding before returning. pad_last_dim checks
    that the last dimension is odd before the actual paddding

    On valid mode the filters must be smaller than the input.

    Parameters
    ----------
    input
        (b, ic, i0, i1, i2).
    filters
        (oc, ic, f0, f1, i2).
    border_mode : {'valid', 'full'}.
    pad_last_dim
        Unconditionally pad the last dimension of the input
        to to turn it from odd to even.  Will strip the
        padding before returning the result.

    """
    if image_shape is None:
        image_shape = input.shape

    if filter_shape is None:
        filter_shape = filters.shape

    b, ic, i0, i1, i2 = image_shape
    oc, ic_, f0, f1, f2 = filter_shape

    is_odd = T.eq(T.mod(input.shape[4], 2), 1)

    if border_mode == 'valid':
        o0 = i0
        o1 = i1
        o2 = i2
        input_padded = input
        if pad_last_dim:
            o2 = ifelse(is_odd, o2 + 1, o2)
            input_padded = T.zeros((b, ic, o0, o1, o2), dtype='float32')
            input_padded = T.set_subtensor(input_padded[:, :, :i0, :i1, :i2],
                                           input)
        filters_padded = T.zeros((oc, ic, o0, o1, o2), dtype='float32')
        filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1, :f2],
                                         filters)

    elif border_mode == 'full':

        o0 = i0 + 2 * (f0 - 1)
        o1 = i1 + 2 * (f1 - 1)
        o2 = i2 + 2 * (f2 - 1)

        if pad_last_dim:
            o2 = ifelse(is_odd, o2 + 1, o2)


        filters_padded = T.zeros((oc, ic, o0, o1, o2), dtype='float32')
        filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1, :f2],
                                         filters)

        input_padded = T.zeros((b, ic, o0, o1, o2), dtype='float32')
        input_padded = T.set_subtensor(input_padded[:, :, (f0 - 1):(f0 - 1 + i0), (f1 - 1):(f1 - 1 + i1), (f2 - 1):(f2 - 1 + i2)],
                                       input)
    else:
        raise ValueError('invalid mode')

    input_flat = input_padded.reshape((b * ic, o0, o1, o2))
    filters_flat = filters_padded.reshape((oc * ic, o0, o1, o2))

    input_fft_flat = cufft(input_flat)  # (b * ic, o0, o1, o2//2 + 1, 2)
    filters_fft_flat = cufft(filters_flat)  # (oc * ic, o0, o1, o2//2 + 1, 2)

    input_fft_v_shape = (b, ic, o0 * o1, o2 // 2 + 1, 2)
    filters_fft_v_shape = (oc, ic, o0 * o1, o2 // 2 + 1, 2)

    input_fft_v = input_fft_flat.reshape(input_fft_v_shape)
    filters_fft_v = filters_fft_flat.reshape(filters_fft_v_shape)

    output_fft_s = mult_and_reduce(input_fft_v, filters_fft_v,
                                   input_shape=input_fft_v_shape,
                                   filter_shape=filters_fft_v_shape)

    output_fft_flat = output_fft_s.reshape((b * oc, o0, o1, o2 // 2 + 1, 2))

    output_flat = cuifft(output_fft_flat)  # (b * oc, o0, o1, o2)

    output_circ = output_flat.reshape((b, oc, o0, o1, o2))  # circular!

    if border_mode == 'valid':
        output = output_circ[:, :, (f0-1):(f0-1 + i0-f0+1), (f1-1):(f1-1 + i1-f1+1), (f2-1):(f2-1 + i2-f2+1)]
    elif border_mode == 'full':
        output = output_circ[:, :, (f0-1):(f0-1 + i0+f0-1), (f1-1):(f1-1 + i1+f1-1), (f2-1):(f2-1 + i2+f2-1)]
    else:
        raise ValueError('invalid mode')

    output = (1.0 / T.cast(o0 * o1 * o2, 'float32')) * output

    return basic_ops.as_cuda_ndarray_variable(output)
