from __future__ import absolute_import, print_function, division
import errno
import logging
import os
import sys
import warnings

import numpy

import theano
from theano import config
from theano.compat import reload
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule


_logger = logging.getLogger('theano.scan_module.scan_perform')


version = 0.293  # must match constant returned in function get_version()

need_reload = False


def try_import():
    global scan_perform
    sys.path[0:0] = [config.compiledir]
    import scan_perform
    del sys.path[0]


def try_reload():
    sys.path[0:0] = [config.compiledir]
    reload(scan_perform)
    del sys.path[0]

try:
    try_import()
    need_reload = True
    if version != getattr(scan_perform, '_version', None):
        raise ImportError()
except ImportError:
    get_lock()
    try:
        try:
            if need_reload:
                try_reload()
            else:
                try_import()
                need_reload = True
            if version != getattr(scan_perform, '_version', None):
                raise ImportError()
        except ImportError:
            if not theano.config.cxx:
                raise ImportError("no c compiler, can't compile cython code")
            _logger.info("Compiling C code for scan")
            dirname = 'scan_perform'
            cfile = os.path.join(theano.__path__[0], 'scan_module',
                                 'scan_perform.c')
            if not os.path.exists(cfile):
                warnings.warn(
                    "The file scan_perform.c is not available. This do"
                    "not happen normally. You are probably in a strange"
                    "setup. This mean Theano can not use the cython code for "
                    "scan. If you"
                    "want to remove this warning, use the Theano flag"
                    "'cxx=' (set to an empty string) to disable all c"
                    "code generation."
                )
                raise ImportError("The file lazylinker_c.c is not available.")

            with open(cfile) as f:
                code = f.read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                try:
                    os.mkdir(loc)
                except OSError as e:
                    assert e.errno == errno.EEXIST
                    assert os.path.exists(loc)

            preargs = ['-fwrapv', '-O2', '-fno-strict-aliasing']
            preargs += cmodule.GCC_compiler.compile_args()
            if False:
                preargs.remove('-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION')
            else:
                numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]
                if bool(numpy_ver >= [1, 7]):
                    preargs.append("-D NPY_ENSUREARRAY=NPY_ARRAY_ENSUREARRAY")
                    preargs.append("-D NPY_ENSURECOPY=NPY_ARRAY_ENSURECOPY")
                    preargs.append("-D NPY_ALIGNED=NPY_ARRAY_ALIGNED")
                    preargs.append("-D NPY_WRITEABLE=NPY_ARRAY_WRITEABLE")
                    preargs.append("-D NPY_UPDATE_ALL=NPY_ARRAY_UPDATE_ALL")
                    preargs.append("-D NPY_C_CONTIGUOUS=NPY_ARRAY_C_CONTIGUOUS")
                    preargs.append("-D NPY_F_CONTIGUOUS=NPY_ARRAY_F_CONTIGUOUS")

            cmodule.GCC_compiler.compile_str(dirname, code, location=loc,
                                             preargs=preargs,
                                             hide_symbols=False)
            init_py = os.path.join(loc, '__init__.py')
            with open(init_py, 'w') as f:
                f.write('_version = %s\n' % version)
            init_pyc = os.path.join(loc, '__init__.pyc')
            if os.path.isfile(init_pyc):
                os.remove(init_pyc)
            try_import()

            try_reload()
            from scan_perform import scan_perform as scan_c
            assert (scan_perform._version ==
                    scan_c.get_version())
            _logger.info("New version %s", scan_perform._version)
    finally:
        release_lock()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                            message="numpy.ndarray size changed")
    from scan_perform.scan_perform import *
assert version == get_version()
