"""
Test compilation modes
"""
from __future__ import absolute_import, print_function, division
import copy
import unittest

import theano
import theano.tensor as T
from theano.compile import Mode, ProfileMode


class T_bunch_of_modes(unittest.TestCase):

    def test1(self):
        linker_classes_involved = []

        predef_modes = ['FAST_COMPILE', 'FAST_RUN', 'DEBUG_MODE']
        predef_modes.append(ProfileMode())

        if theano.config.cxx:
            linkers = ['py', 'c|py', 'c|py_nogc', 'vm', 'vm_nogc',
                       'cvm', 'cvm_nogc']
        else:
            linkers = ['py', 'c|py', 'c|py_nogc', 'vm', 'vm_nogc']
        modes = predef_modes + [Mode(linker, 'fast_run') for linker in linkers]

        for mode in modes:
            x = T.matrix()
            y = T.vector()
            f = theano.function([x, y], x + y, mode=mode)
            f([[1, 2], [3, 4]], [5, 6])
            linker_classes_involved.append(f.maker.mode.linker.__class__)

        assert 5 == len(set(linker_classes_involved))


class T_ProfileMode_WrapLinker(unittest.TestCase):
    def test_1(self):
        x = T.matrix()
        mode = ProfileMode()
        theano.function([x], x * 2, mode=mode)

        default_mode = theano.compile.mode.get_default_mode()
        modified_mode = default_mode.including('specialize')

        copy.deepcopy(modified_mode)

        linker = theano.compile.mode.get_default_mode().linker
        assert not hasattr(linker, "fgraph") or linker.fgraph is None


if __name__ == '__main__':
    unittest.main()
