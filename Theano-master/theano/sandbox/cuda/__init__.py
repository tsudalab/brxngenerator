from __future__ import absolute_import, print_function, division
import atexit
import errno
import logging
import os
import shutil
import stat
import sys
import warnings

import theano
from theano.compat import get_unbound_function
from theano.compile import optdb
from theano.gof import EquilibriumDB, SequenceDB
from theano.gof.cmodule import get_lib_extension
from theano.gof.compilelock import get_lock, release_lock
from theano import config
from . import nvcc_compiler

from theano.tensor.basic import register_transfer

gpu_optimizer = EquilibriumDB(ignore_newtrees=False)
gpu_seqopt = SequenceDB()


def register_opt(*tags, **kwargs):
    if any([not isinstance(t, str) for t in tags]):
        raise RuntimeError("Bad call to register_opt."
                           " All tags must be strings.", tags)

    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'fast_compile',
                               'gpu', *tags, **kwargs)
        return local_opt
    return f


_logger_name = 'theano.sandbox.cuda'
_logger = logging.getLogger(_logger_name)

nvcc_compiler.is_nvcc_available()


cuda_available = True

cuda_warning_is_displayed = False

cuda_enabled = False


def set_cuda_disabled():
    """
    Function used to disable cuda.

    A warning is displayed, so that the user is aware that cuda-based code is
    not going to work.
    Note that there is no point calling this function from outside of
    `cuda.__init__`, since it has no effect once the module is loaded.

    """
    global cuda_available, cuda_warning_is_displayed
    cuda_available = False

cuda_path = os.path.abspath(os.path.split(__file__)[0])

cuda_ndarray_loc = os.path.join(config.compiledir, 'cuda_ndarray')
cuda_ndarray_so = os.path.join(cuda_ndarray_loc,
                               'cuda_ndarray.' + get_lib_extension())
libcuda_ndarray_so = os.path.join(cuda_ndarray_loc,
                               'libcuda_ndarray.' + get_lib_extension())


def try_import():
    """
    Load the cuda_ndarray module if present and up to date.
    Return True if loaded correctly, otherwise return False.

    """
    cuda_files = (
        'cuda_ndarray.cu',
        'cuda_ndarray.cuh',
        'conv_full_kernel.cu',
        'cnmem.h',
        'cnmem.cpp',
        'conv_kernel.cu')
    stat_times = [os.stat(os.path.join(cuda_path, cuda_file))[stat.ST_MTIME]
                  for cuda_file in cuda_files]
    date = max(stat_times)
    if os.path.exists(cuda_ndarray_so):
        if date >= os.stat(cuda_ndarray_so)[stat.ST_MTIME]:
            return False
    try:
        sys.path[0:0] = [config.compiledir]
        import cuda_ndarray.cuda_ndarray
        del sys.path[0]
    except ImportError:
        return False
    return True


if not nvcc_compiler.is_nvcc_available() or not theano.config.cxx:
    set_cuda_disabled()
    compile_cuda_ndarray = False
elif not config.device.startswith('gpu') and config.force_device:
    set_cuda_disabled()
    compile_cuda_ndarray = False
else:
    nvcc_compiler.add_standard_rpath(cuda_ndarray_loc)
    compile_cuda_ndarray = not try_import()


if compile_cuda_ndarray and cuda_available:
    get_lock()
    try:
        if not try_import():
            try:
                if not nvcc_compiler.is_nvcc_available():
                    set_cuda_disabled()

                if cuda_available:
                    code = open(os.path.join(cuda_path,
                                             "cuda_ndarray.cu")).read()

                    if not os.path.exists(cuda_ndarray_loc):
                        os.makedirs(cuda_ndarray_loc)

                    if 'TMPDIR' in os.environ:
                        tmpdir = os.environ['TMPDIR']
                        if not os.path.exists(tmpdir):
                            os.makedirs(tmpdir)
                    compiler = nvcc_compiler.NVCC_compiler()
                    preargs = ['-O3'] + compiler.compile_args()
                    compiler.compile_str(
                            'cuda_ndarray',
                            code,
                            location=cuda_ndarray_loc,
                            include_dirs=[cuda_path],
                            libs=[config.cublas.lib],
                            preargs=preargs,
                    )
                    from cuda_ndarray.cuda_ndarray import *
            except Exception as e:
                _logger.error("Failed to compile cuda_ndarray.cu: %s", str(e))
                set_cuda_disabled()
    finally:
        release_lock()

del compile_cuda_ndarray

if cuda_available:
    global cuda_initialization_error_message
    from cuda_ndarray.cuda_ndarray import *

    def ok():
        """
        Check if an existing library exists and can be read.

        """
        try:
            open(libcuda_ndarray_so).close()
            return True
        except IOError:
            return False
    if not ok():
        if sys.platform == "win32":
            shutil.copyfile(cuda_ndarray_so, libcuda_ndarray_so)
        else:
            try:
                os.symlink(cuda_ndarray_so, libcuda_ndarray_so)
            except OSError as e:
                if getattr(e, 'errno', None) != errno.EEXIST or not ok():
                    raise
    try:
        gpu_init()
        cuda_available = True
        cuda_initialization_error_message = ""
        atexit.register(gpu_shutdown)
    except EnvironmentError as e:
        cuda_available = False
        cuda_initialization_error_message = " ".join(e.args)
else:
    cuda_initialization_error_message = 'cuda unavailable'


class GpuOp(theano.gof.Op):

    """
    Parent class for all GPU Ops.

    This class ensures we verify the GPU is working properly when a GPU Op is
    used for the first time.

    It is defined in __init__.py so that it exists even when `cuda_available`
    is False (this is necessary to avoid breaking the test suite).

    """

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        if use.device_number is None:
            use("gpu",
                force=True,
                default_to_move_computation_to_gpu=False,
                move_shared_float32_to_gpu=False,
                enable_cuda=False)
        return super(GpuOp, self).make_thunk(node, storage_map,
                                             compute_map, no_recycling)

theano.compile.debugmode.default_make_thunk.append(
    get_unbound_function(GpuOp.make_thunk))

from theano.sandbox.cuda.var import (CudaNdarrayVariable,
                                     CudaNdarrayConstant,
                                     CudaNdarraySharedVariable,
                                     float32_shared_constructor)
from theano.sandbox.cuda.type import CudaNdarrayType


def dnn_available():
    if config.dnn.enabled == "False":
        dnn_available.avail = False
        dnn_available.msg = "disabled by dnn.enabled flag"
    if dnn_available.avail is None and not cuda_available:
        dnn_available.msg = "CUDA not available"
        dnn_available.avail = False
    elif dnn_available.avail is None:
        dev = active_device_number()
        if device_properties(dev)['major'] < 3:
            dnn_available.msg = "Device not supported by cuDNN"
            dnn_available.avail = False
        else:
            preambule = """
            """

            body = """
cudnnHandle_t _handle = NULL;
cudnnStatus_t err;
if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
  fprintf(stderr, "could not create cuDNN handle: %s",
          cudnnGetErrorString(err));
  return 1;
}
"""
            params = ["-l", "cudnn", "-I" + os.path.dirname(__file__)]
            if config.dnn.include_path:
                params.append("-I" + config.dnn.include_path)
            if config.dnn.library_path:
                params.append("-L" + config.dnn.library_path)
            if config.nvcc.compiler_bindir:
                params.extend(['--compiler-bindir',
                               config.nvcc.compiler_bindir])
            comp, out, err = nvcc_compiler.NVCC_compiler.try_flags(
                flag_list=params, preambule=preambule, body=body,
                try_run=False, output=True)

            dnn_available.avail = comp
            if not dnn_available.avail:
                dnn_available.msg = (
                    "Theano can not compile with cuDNN. We got this error:\n" +
                    str(err))
            else:
                v = dnn_version()
                if isinstance(v, tuple) and v[0] != v[1]:
                    dnn_available.avail = False
                    dnn_available.msg = ("Mixed dnn version. The header is"
                                         " from one version, but we link with"
                                         " a different version %s" % str(v))
                    raise RuntimeError(dnn_available.msg)
                if v == -1 or v[0] < 3007:
                    dnn_available.avail = False
                    dnn_available.msg = (
                        "You have an old release of CuDNN (or a release "
                        "candidate) that isn't supported.  Please update to "
                        "at least v3 final version.")
                    raise RuntimeError(dnn_available.msg)
    if config.dnn.enabled == "True":
        if not dnn_available.avail:
            raise RuntimeError(
                "You enabled CuDNN, but we aren't able to use it: %s" %
                dnn_available.msg)
    return dnn_available.avail


dnn_available.avail = None
dnn_available.msg = None


class DnnVersion(GpuOp):
    def c_compiler(self):
        return nvcc_compiler.NVCC_compiler

    def c_headers(self):
        return ['cudnn.h']

    def c_header_dirs(self):
        return [config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def c_compile_args(self):
        return ['-Wl,-rpath,' + config.dnn.library_path]

    def c_support_code(self):
        return """
"""

    def make_node(self):
        return theano.gof.Apply(self, [], [theano.gof.Generic()()])

    def c_code(self, node, name, inputs, outputs, sub):
        o = outputs[0]
        return """
        %(o)s = PyTuple_Pack(2, PyInt_FromLong(CUDNN_VERSION), PyInt_FromLong(cudnnGetVersion()));
        %(o)s = PyInt_FromLong(-1);
        """ % locals()

    def do_constant_folding(self, node):
        return False

    def c_code_cache_version(self):
        return None


def dnn_version():
    """Return the current cuDNN version we compile with.

    This returns a tuple with the header version and the library
    version we link with. For older cudnn version without version
    information, we return -1.

    """
    if not dnn_available():
        raise Exception(
            "We can't determine the cudnn version as it is not available",
            dnn_available.msg)

    if dnn_version.v is None:
        f = theano.function([], DnnVersion()(),
                            theano.Mode(optimizer=None),
                            profile=False)
        dnn_version.v = f()
    return dnn_version.v
dnn_version.v = None


if cuda_available:
    import cuda_ndarray.cuda_ndarray
    if cuda_ndarray_so != cuda_ndarray.cuda_ndarray.__file__:
        _logger.warning("cuda_ndarray was loaded from %s, but Theano expected "
                "to load it from %s. This is not expected as theano should "
                "compile it automatically for you. Do you have a directory "
                "called cuda_ndarray in your LD_LIBRARY_PATH environment "
                "variable? If so, please remove it as it is outdated.",
                cuda_ndarray.cuda_ndarray.__file__,
                cuda_ndarray_so)

    shared_constructor = float32_shared_constructor

    from . import basic_ops
    from .basic_ops import (
            GpuFromHost, HostFromGpu, GpuElemwise,
            GpuDimShuffle, GpuCAReduce, GpuReshape, GpuContiguous,
            GpuSubtensor, GpuIncSubtensor,
            GpuAdvancedSubtensor1, GpuAdvancedIncSubtensor1,
            gpu_flatten, GpuFlatten, GpuShape, GpuAlloc, GpuAllocEmpty, GpuSplit,
            GpuJoin, fscalar, fvector, fmatrix, frow, fcol,
            ftensor3, ftensor4,
            scalar, vector, matrix, row, col,
            tensor3, tensor4)
    from .basic_ops import (host_from_gpu, gpu_from_host,
            as_cuda_array, as_cuda_ndarray_variable)
    import cuda_ndarray
    from . import opt, dnn
    from .rng_curand import CURAND_RandomStreams

    def transfer(x, target):
        if target == 'gpu':
            return as_cuda_ndarray_variable(x)

    register_transfer(transfer)


def use(device,
        force=False,
        default_to_move_computation_to_gpu=True,
        move_shared_float32_to_gpu=True,
        enable_cuda=True,
        test_driver=True):
    """
    Error and warning about CUDA should be displayed only when this
    function is called. We need to be able to load this module only
    to check if it is available!

    Parameters
    ----------
    device : string
        "cpu", "gpu", "gpuN" (N is the device number to use).
    force
        Will always raise an exception if we can't use the gpu.
    default_to_move_computation_to_gpu
        If gpu init succeeded, enable by default optimizations to move
        computations to the gpu.
    move_shared_float32_to_gpu
        If gpu init succeeded, put new shared variables in float32 on the gpu.
    enable_cuda
        If the gpu is correctly enabled, set the variable cuda_enabled to True.

    """
    global cuda_enabled, cuda_initialization_error_message
    if force and not cuda_available and device.startswith('gpu'):
        if not nvcc_compiler.is_nvcc_available():
            raise EnvironmentError("You forced the use of gpu device '%s', but"
                                   " nvcc was not found. Set it in your PATH "
                                   "environment variable or set the Theano "
                                   "flags 'cuda.root' to its directory"
                                   "" % device)
        else:
            raise EnvironmentError("You forced the use of gpu device %s, "
                                   "but CUDA initialization failed "
                                   "with error:\n%s" % (
                device, cuda_initialization_error_message))
    elif not nvcc_compiler.is_nvcc_available():
        _logger.error('nvcc compiler not found on $PATH.'
              ' Check your nvcc installation and try again.')
        return
    elif not cuda_available:
        error_addendum = ""
        try:
            if cuda_initialization_error_message:
                error_addendum = (" (error: %s)" %
                                  cuda_initialization_error_message)
        except NameError:
            pass
        _logger.warning('CUDA is installed, but device %s is not available %s',
                device, error_addendum)
        return

    if device == 'gpu':
        pass
    elif device.startswith('gpu'):
        device = int(device[3:])
    elif device == 'cpu':
        device = -1
    else:
        raise ValueError("Invalid device identifier", device)
    if use.device_number is None:
        if device != 'gpu' and device < 0:
            return

        pycuda_init_dev = False
        if config.pycuda.init:
            import theano.misc.pycuda_init
            pycuda_init_dev = theano.misc.pycuda_init.pycuda_available

        try:
            if pycuda_init_dev:
                use.device_number = active_device_number()
                gpu_init(use.device_number, config.lib.cnmem)
            elif(device != 'gpu'):
                assert isinstance(device, int)
                gpu_init(device, config.lib.cnmem)
                use.device_number = device
                active_device = active_device_number()
                assert active_device == device, (active_device, device)
            else:
                if not hasattr(cuda_ndarray.cuda_ndarray, 'select_a_gpu'):
                    raise Exception(
                        "Delete your Theano cache. The automatic"
                        " recompilation did not work.")
                cuda_ndarray.cuda_ndarray.select_a_gpu()
                use.device_number = active_device_number()
                gpu_init(use.device_number, config.lib.cnmem)

            if test_driver:
                import theano.sandbox.cuda.tests.test_driver
                theano.sandbox.cuda.tests.test_driver.test_nvidia_driver1()
            if device_properties(use.device_number)["warpSize"] != 32:
                raise ValueError("Your GPU has a warpSize != 32. Currently"
                                 " we have code that depends on this. Email"
                                 " the Theano mailing list to tell us about"
                                 " this new GPU as we don't know any with"
                                 " this property")

            if config.print_active_device:
                if config.lib.cnmem:
                    if config.lib.cnmem > 1:
                        cnmem_enabled = "enabled with initial size: %d MB" % config.lib.cnmem
                    else:
                        cnmem = min(config.lib.cnmem, 0.95) * 100
                        cnmem_enabled = "enabled with initial size: %.1f%% of memory" % cnmem
                else:
                    cnmem_enabled = "disabled"
                cudnn_version = "not available"
                warn = None
                try:
                    if dnn_available():
                        (hdr_v, runtime_v) = dnn_version()
                        cudnn_version = runtime_v
                        if cudnn_version > 4100:
                            warn = ("Your CuDNN version is more recent then Theano."
                                    " If you see problems, try updating Theano or"
                                    " downgrading CuDNN to version 4.")
                except Exception:
                    pass
                print("Using gpu device %d: %s (CNMeM is %s, CuDNN %s)" % (
                    active_device_number(),
                    active_device_name(),
                    cnmem_enabled,
                    cudnn_version,),
                      file=sys.stderr)
                if warn:
                    warnings.warn(warn)

            if device_properties(use.device_number)['regsPerBlock'] < 16384:
                _logger.warning(
                        "You are probably using an old GPU, that Theano"
                        " does not support."
                        " This means GPU code will most likely be slow AND may"
                        " crash when we try to use features"
                        " that your GPU does not support.")

        except (EnvironmentError, ValueError, RuntimeError) as e:
            _logger.error(("ERROR: Not using GPU."
                           " Initialisation of device %s failed:\n%s"),
                          str(device), e)
            cuda_enabled = False
            if force:
                e.args += (("You asked to force this device and it failed."
                            " No fallback to the cpu or other gpu device."),)
                raise

    elif use.device_number != device and device != 'gpu':
        _logger.warning(("Ignoring call to use(%s), GPU number %i "
            "is already in use."),
            str(device), use.device_number)

    if move_shared_float32_to_gpu:
        handle_shared_float32(True)

    if enable_cuda:
        cuda_enabled = True

    if default_to_move_computation_to_gpu:
        optdb.add_tags('gpu_opt',
                       'fast_compile',
                       'fast_run')
        optdb.add_tags('gpu_after_fusion',
                       'fast_run')
        optdb.add_tags('gpu_scanOp_make_inplace',
                       'fast_run')

    if force:
        try:
            cuda_ndarray.cuda_ndarray.CudaNdarray.zeros((5, 5))
        except (Exception, NameError) as e:
            e.args += ("ERROR: GPU forced but failed. ",)
            raise
use.device_number = None


def unuse():
    """
    This undo what was done by the call to.

    use('gpu[0-9]', default_to_move_computation_to_gpu=True,
        move_shared_float32_to_gpu=True,
        enable_cuda=True)

    This is used in Pylearn2 tests to enable/disable the GPU when needed.

    After this call, the rest of Theano think the GPU shouldn't be used by
    default.

    """
    global cuda_enabled
    cuda_enabled = False
    handle_shared_float32(False)
    optdb.remove_tags('gpu_opt',
                      'fast_compile',
                      'fast_run')
    optdb.remove_tags('gpu_after_fusion',
                      'fast_run')


def handle_shared_float32(tf):
    """
    Set the default shared type for float32 tensor to CudaNdarrayType.

    This function is intended to be called from use(gpu_index), not directly.

    """
    if tf:
        theano.compile.shared_constructor(float32_shared_constructor)
    else:
        theano.compile.shared_constructor(float32_shared_constructor, True)
        assert (float32_shared_constructor not in
                theano.compile.shared.constructors)

if config.device.startswith('gpu'):
    use(device=config.device, force=config.force_device, test_driver=False)
elif config.init_gpu_device.startswith('gpu'):
    assert config.device == "cpu", (
        "We can use the Theano flag init_gpu_device"
        " only when the Theano flag device=='cpu'")
    _logger.warning(("GPU device %s will be initialized, and used if a GPU is "
          "needed. "
          "However, no computation, nor shared variables, will be implicitly "
          "moved to that device. If you want that behavior, use the 'device' "
          "flag instead."),
          config.init_gpu_device)
    use(device=config.init_gpu_device,
        force=config.force_device,
        default_to_move_computation_to_gpu=False,
        move_shared_float32_to_gpu=False,
        enable_cuda=False, test_driver=False)
