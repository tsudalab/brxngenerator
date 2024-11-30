from __future__ import absolute_import, print_function, division
import errno
import logging
import os
from six.moves import reload_module as reload
import sys
import warnings


import theano
from theano import config
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule

_logger = logging.getLogger('theano.gof.lazylinker_c')

force_compile = False
version = 0.21  # must match constant returned in function get_version()
lazylinker_ext = None


def try_import():
    global lazylinker_ext
    sys.path[0:0] = [config.compiledir]
    import lazylinker_ext  # noqa
    del sys.path[0]


def try_reload():
    sys.path[0:0] = [config.compiledir]
    reload(lazylinker_ext)
    del sys.path[0]

try:
    location = os.path.join(config.compiledir, 'lazylinker_ext')
    if not os.path.exists(location):
        try:
            os.mkdir(location)
        except OSError as e:
            assert e.errno == errno.EEXIST
            assert os.path.isdir(location)

    init_file = os.path.join(location, '__init__.py')
    if not os.path.exists(init_file):
        try:
            open(init_file, 'w').close()
        except IOError as e:
            if os.path.exists(init_file):
                pass  # has already been created
            else:
                e.args += ('%s exist? %s' % (location,
                                             os.path.exists(location)),)
                raise

    _need_reload = False
    if force_compile:
        raise ImportError()
    else:
        try_import()
        _need_reload = True
        if version != getattr(lazylinker_ext, '_version', None):
            raise ImportError()
except ImportError:
    get_lock()
    try:
        try:
            if force_compile:
                raise ImportError()
            if _need_reload:
                try_reload()
            else:
                try_import()
                _need_reload = True
            if version != getattr(lazylinker_ext, '_version', None):
                raise ImportError()
        except ImportError:
            if not theano.config.cxx:
                raise
            _logger.info("Compiling new CVM")
            dirname = 'lazylinker_ext'
            cfile = os.path.join(theano.__path__[0], 'gof', 'lazylinker_c.c')
            if not os.path.exists(cfile):
                warnings.warn(
                    "The file lazylinker_c.c is not available. This do"
                    "not happen normally. You are probably in a strange"
                    "setup. This mean Theano can not use the cvm:"
                    "our c execution engine for Theano function. If you"
                    "want to remove this warning, use the Theano flag"
                    "'cxx=' (set to an empty string) to disable all c"
                    "code generation."
                )
                raise ImportError("The file lazylinker_c.c is not available.")
            code = open(cfile).read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                try:
                    os.mkdir(loc)
                except OSError as e:
                    assert e.errno == errno.EEXIST
                    assert os.path.exists(loc)

            args = cmodule.GCC_compiler.compile_args()
            cmodule.GCC_compiler.compile_str(dirname, code, location=loc,
                                             preargs=args)
            init_py = os.path.join(loc, '__init__.py')
            open(init_py, 'w').write('_version = %s\n' % version)
            init_pyc = os.path.join(loc, '__init__.pyc')
            if os.path.isfile(init_pyc):
                os.remove(init_pyc)
            try_import()
            try_reload()
            from lazylinker_ext import lazylinker_ext as lazy_c
            assert (lazylinker_ext._version ==
                    lazy_c.get_version())
            _logger.info("New version %s", lazylinker_ext._version)
    finally:
        release_lock()

from lazylinker_ext.lazylinker_ext import *  # noqa
assert force_compile or (version == get_version())
