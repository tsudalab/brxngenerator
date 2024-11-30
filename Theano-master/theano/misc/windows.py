from __future__ import absolute_import, print_function, division
import os
import subprocess


def subprocess_Popen(command, **params):
    """
    Utility function to work around windows behavior that open windows.

    :see: call_subprocess_Popen and output_subprocess_Popen
    """
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        try:
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        except AttributeError:
            startupinfo.dwFlags |= subprocess._subprocess.STARTF_USESHOWWINDOW

        params['shell'] = True

    stdin = None
    if "stdin" not in params:
        stdin = open(os.devnull)
        params['stdin'] = stdin.fileno()

    try:
        proc = subprocess.Popen(command, startupinfo=startupinfo, **params)
    finally:
        if stdin is not None:
            del stdin
    return proc


def call_subprocess_Popen(command, **params):
    """
    Calls subprocess_Popen and discards the output, returning only the
    exit code.
    """
    if 'stdout' in params or 'stderr' in params:
        raise TypeError("don't use stderr or stdout with call_subprocess_Popen")
    with open(os.devnull, 'wb') as null:
        params.setdefault('stdin', null)
        params['stdout'] = null
        params['stderr'] = null
        p = subprocess_Popen(command, **params)
        returncode = p.wait()
    return returncode


def output_subprocess_Popen(command, **params):
    """
    Calls subprocess_Popen, returning the output, error and exit code
    in a tuple.
    """
    if 'stdout' in params or 'stderr' in params:
        raise TypeError("don't use stderr or stdout with output_subprocess_Popen")
    if not hasattr(params, 'stdin'):
        null = open(os.devnull, 'wb')
        params['stdin'] = null
    params['stdout'] = subprocess.PIPE
    params['stderr'] = subprocess.PIPE
    p = subprocess_Popen(command, **params)
    out = p.communicate()
    return out + (p.returncode,)
