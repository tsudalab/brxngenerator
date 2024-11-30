"""
Test config options.
"""
from __future__ import absolute_import, print_function, division
import unittest
from theano.configparser import AddConfigVar, ConfigParam, THEANO_FLAGS_DICT


class T_config(unittest.TestCase):

    def test_invalid_default(self):

        def filter(val):
            if val == 'invalid':
                raise ValueError()
            else:
                return val

        try:
            AddConfigVar(
                'T_config.test_invalid_default_a',
                doc='unittest',
                configparam=ConfigParam('invalid', filter=filter),
                in_c_key=False)
            assert False
        except ValueError:
            pass

        THEANO_FLAGS_DICT['T_config.test_invalid_default_b'] = 'ok'
        AddConfigVar('T_config.test_invalid_default_b',
                     doc='unittest',
                     configparam=ConfigParam('invalid', filter=filter),
                     in_c_key=False)

        assert 'T_config.test_invalid_default_b' not in THEANO_FLAGS_DICT

