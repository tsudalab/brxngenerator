from __future__ import absolute_import, print_function, division
import os
import unittest
import sys

from numpy.testing.nosetester import NoseTester


class TheanoNoseTester(NoseTester):
    """
    Nose test runner.

    This class enables running nose tests from inside Theano,
    by calling theano.test().
    This version is more adapted to what we want than Numpy's one.
    """

    def _test_argv(self, verbose, extra_argv):
        """
        Generate argv for nosetest command

        :type verbose: int
        :param verbose: Verbosity value for test outputs, in the range 1-10.
                        Default is 1.

        :type extra_argv: list
        :param extra_argv: List with any extra arguments to pass to nosetests.
        """
        argv = [__file__, self.package_path]
        argv += ['--verbosity', str(verbose)]
        if extra_argv:
            argv += extra_argv
        return argv

    def _show_system_info(self):
        import theano
        print("Theano version %s" % theano.__version__)
        theano_dir = os.path.dirname(theano.__file__)
        print("theano is installed in %s" % theano_dir)

        super(TheanoNoseTester, self)._show_system_info()

    def prepare_test_args(self, verbose=1, extra_argv=None, coverage=False,
                          capture=True, knownfailure=True):
        """
        Prepare arguments for the `test` method.

        Takes the same arguments as `test`.
        """
        import nose.plugins.builtin
        argv = self._test_argv(verbose, extra_argv)

        if coverage:
            argv += ['--cover-package=%s' % self.package_name,
                     '--with-coverage', '--cover-tests',
                     '--cover-inclusive', '--cover-erase']

        if not capture:
            argv += ['-s']

        plugins = []
        if knownfailure:
            from numpy.testing.noseclasses import KnownFailure
            plugins.append(KnownFailure())
        plugins += [p() for p in nose.plugins.builtin.plugins]

        return argv, plugins

    def test(self, verbose=1, extra_argv=None, coverage=False, capture=True,
             knownfailure=True):
        """
        Run tests for module using nose.

        :type verbose: int
        :param verbose: Verbosity value for test outputs, in the range 1-10.
                        Default is 1.

        :type extra_argv: list
        :param extra_argv: List with any extra arguments to pass to nosetests.

        :type coverage: bool
        :param coverage: If True, report coverage of Theano
                         code. Default is False.

        :type capture: bool
        :param capture: If True, capture the standard output of the tests, like
                        nosetests does in command-line. The output of failing
                        tests will be displayed at the end. Default is True.

        :type knownfailure: bool
        :param knownfailure: If True, tests raising KnownFailureTest will
                not be considered Errors nor Failure, but reported as
                "known failures" and treated quite like skipped tests.
                Default is True.

        :returns: Returns the result of running the tests as a
                  ``nose.result.TextTestResult`` object.
        """
        from nose.config import Config
        from nose.plugins.manager import PluginManager
        from numpy.testing.noseclasses import NumpyTestProgram
        if not os.path.exists('theano/__init__.py'):
            try:
                from theano import config
                if config.device != "cpu":
                    raise ValueError("Theano tests must be run with device=cpu."
                                     " This will also run GPU tests when possible.\n"
                                     " If you want GPU-related tests to run on a"
                                     " specific GPU device, and not the default one,"
                                     " you should use the init_gpu_device theano flag.")
            except ImportError:
                pass

        verbose = min(verbose, 3)
        self._show_system_info()

        cwd = os.getcwd()
        if self.package_path in os.listdir(cwd):
            raise RuntimeError((
                "This function does not run correctly when, at the time "
                "theano was imported, the working directory was theano's "
                "parent directory. You should exit your Python prompt, change "
                "directory, then launch Python again, import theano, then "
                "launch theano.test()."))

        argv, plugins = self.prepare_test_args(verbose, extra_argv, coverage,
                                               capture, knownfailure)

        cfg = Config(includeExe=True, plugins=PluginManager(plugins=plugins))
        t = NumpyTestProgram(argv=argv, exit=False, config=cfg)
        return t.result


def main(modulename):
    if 0:
        unittest.main()
    elif len(sys.argv) == 2 and sys.argv[1] == "--debug":
        module = __import__(modulename)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        tests.debug()
    elif len(sys.argv) == 1:
        module = __import__(modulename)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        unittest.TextTestRunner(verbosity=2).run(tests)
    else:
        print("options: [--debug]")
