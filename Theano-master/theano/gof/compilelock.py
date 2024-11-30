from __future__ import absolute_import, print_function, division

import atexit
import os
import socket  # only used for gethostname()
import time
import logging

from contextlib import contextmanager

import numpy as np

from theano import config

random = np.random.RandomState([2015, 8, 2])

_logger = logging.getLogger("theano.gof.compilelock")
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.INFO)

hostname = socket.gethostname()


def force_unlock():
    """
    Delete the compilation lock if someone else has it.

    """
    get_lock(min_wait=0, max_wait=0.001, timeout=0)
    release_lock()


@contextmanager
def lock_ctx(lock_dir=None, keep_lock=False, **kw):
    get_lock(lock_dir=lock_dir, **kw)
    yield
    if not keep_lock:
        release_lock()


def _get_lock(lock_dir=None, **kw):
    """
    Obtain lock on compilation directory.

    Parameters
    ----------
    kw
        Additional arguments to be forwarded to the `lock` function when
        acquiring the lock.

    Notes
    -----
    We can lock only on 1 directory at a time.

    """
    if lock_dir is None:
        lock_dir = os.path.join(config.compiledir, 'lock_dir')
    if not hasattr(get_lock, 'n_lock'):
        get_lock.n_lock = 0
        if not hasattr(get_lock, 'lock_is_enabled'):
            get_lock.lock_is_enabled = True
        get_lock.lock_dir = lock_dir
        get_lock.unlocker = Unlocker(get_lock.lock_dir)
    else:
        if lock_dir != get_lock.lock_dir:
            assert get_lock.n_lock == 0
            get_lock.lock_dir = lock_dir
            get_lock.unlocker = Unlocker(get_lock.lock_dir)

    if get_lock.lock_is_enabled:
        if get_lock.n_lock == 0:
            lock(get_lock.lock_dir, **kw)
            atexit.register(Unlocker.unlock, get_lock.unlocker)
            get_lock.start_time = time.time()
        else:
            if get_lock.start_time is None:
                while get_lock.n_lock > 0:
                    release_lock()
                raise Exception("For some unknow reason, the lock was already "
                                "taken, but no start time was registered.")
            now = time.time()
            if now - get_lock.start_time > config.compile.timeout / 2:
                lockpath = os.path.join(get_lock.lock_dir, 'lock')
                _logger.info('Refreshing lock %s', str(lockpath))
                refresh_lock(lockpath)
                get_lock.start_time = now
    get_lock.n_lock += 1


get_lock = _get_lock


def release_lock():
    """
    Release lock on compilation directory.

    """
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock(force=False)


def set_lock_status(use_lock):
    """
    Enable or disable the lock on the compilation directory (which is enabled
    by default). Disabling may make compilation slightly faster (but is not
    recommended for parallel execution).

    Parameters
    ----------
    use_lock : bool
        Whether to use the compilation lock or not.

    """
    get_lock.lock_is_enabled = use_lock

notset = object()


def lock(tmp_dir, timeout=notset, min_wait=None, max_wait=None, verbosity=1):
    """
    Obtain lock access by creating a given temporary directory (whose base will
    be created if needed, but will not be deleted after the lock is removed).
    If access is refused by the same lock owner during more than 'timeout'
    seconds, then the current lock is overridden. If timeout is None, then no
    timeout is performed.

    The lock is performed by creating a 'lock' file in 'tmp_dir' that contains
    a unique id identifying the owner of the lock (the process id, followed by
    a random string).

    When there is already a lock, the process sleeps for a random amount of
    time between min_wait and max_wait seconds before trying again.

    If 'verbosity' is >= 1, then a message will be displayed when we need to
    wait for the lock. If it is set to a value >1, then this message will be
    displayed each time we re-check for the presence of the lock. Otherwise it
    is displayed only when we notice the lock's owner has changed.

    Parameters
    ----------
    tmp_dir : str
        Lock directory that will be created when acquiring the lock.
    timeout : int or None
        Time (in seconds) to wait before replacing an existing lock (default
        config 'compile.timeout').
    min_wait: int
        Minimum time (in seconds) to wait before trying again to get the lock
        (default config 'compile.wait').
    max_wait: int
        Maximum time (in seconds) to wait before trying again to get the lock
        (default 2 * min_wait).
    verbosity : int
        Amount of feedback displayed to screen (default 1).

    """
    if min_wait is None:
        min_wait = config.compile.wait
    if max_wait is None:
        max_wait = min_wait * 2
    if timeout is notset:
        timeout = config.compile.timeout
    base_lock = os.path.dirname(tmp_dir)
    if not os.path.isdir(base_lock):
        try:
            os.makedirs(base_lock)
        except OSError:
            time.sleep(2)
    assert os.path.isdir(base_lock)

    lock_file = os.path.join(tmp_dir, 'lock')
    my_pid = os.getpid()
    no_display = (verbosity == 0)

    nb_error = 0
    nb_wait = 0
    while True:
        try:
            last_owner = 'no_owner'
            time_start = time.time()
            other_dead = False
            while os.path.isdir(tmp_dir):
                try:
                    with open(lock_file) as f:
                        read_owner = f.readlines()[0].strip()

                    try:
                        other_host = read_owner.split('_')[2]
                    except IndexError:
                        other_host = ()  # make sure it isn't equal to any host
                    if other_host == hostname:
                        try:
                            os.kill(int(read_owner.split('_')[0]), 0)
                        except OSError:
                            other_dead = True
                        except AttributeError:
                            pass  # os.kill does not exist on windows
                except Exception:
                    read_owner = 'failure'
                if other_dead:
                    if not no_display:
                        msg = "process '%s'" % read_owner.split('_')[0]
                        _logger.warning("Overriding existing lock by dead %s "
                                        "(I am process '%s')", msg, my_pid)
                    get_lock.unlocker.unlock(force=True)
                    continue
                if last_owner == read_owner:
                    if (timeout is not None and
                            time.time() - time_start >= timeout):
                        if not no_display:
                            if read_owner == 'failure':
                                msg = 'unknown process'
                            else:
                                msg = "process '%s'" % read_owner.split('_')[0]
                            _logger.warning("Overriding existing lock by %s "
                                            "(I am process '%s')", msg, my_pid)
                        get_lock.unlocker.unlock(force=True)
                        continue
                else:
                    last_owner = read_owner
                    time_start = time.time()
                    no_display = (verbosity == 0)
                if not no_display and nb_wait > 0:
                    if read_owner == 'failure':
                        msg = 'unknown process'
                    else:
                        msg = "process '%s'" % read_owner.split('_')[0]
                    _logger.info("Waiting for existing lock by %s (I am "
                                 "process '%s')", msg, my_pid)
                    _logger.info("To manually release the lock, delete %s",
                                 tmp_dir)
                    if verbosity <= 1:
                        no_display = True
                nb_wait += 1
                time.sleep(random.uniform(min_wait, max_wait))

            try:
                os.mkdir(tmp_dir)
            except OSError:
                nb_error += 1
                if nb_error < 10:
                    continue
                else:
                    raise
            assert os.path.isdir(tmp_dir)

            unique_id = refresh_lock(lock_file)

            with open(lock_file) as f:
                owner = f.readlines()[0].strip()

            if owner != unique_id:
                continue
            else:
                return

        except Exception as e:
            _logger.warning("Something wrong happened: %s %s", type(e), e)
            nb_error += 1
            if nb_error > 10:
                raise
            time.sleep(random.uniform(min_wait, max_wait))
            continue


def refresh_lock(lock_file):
    """
    'Refresh' an existing lock by re-writing the file containing the owner's
    unique id, using a new (randomly generated) id, which is also returned.

    """
    unique_id = '%s_%s_%s' % (
        os.getpid(),
        ''.join([str(random.randint(0, 9)) for i in range(10)]),
        hostname)
    try:
        with open(lock_file, 'w') as lock_write:
            lock_write.write(unique_id + '\n')
    except Exception:
        while get_lock.n_lock > 0:
            release_lock()
        _logger.warn('Refreshing lock failed, we release the'
                     ' lock before raising again the exception')
        raise
    return unique_id


class Unlocker(object):
    """
    Class wrapper around release mechanism so that the lock is automatically
    released when the program exits (even when crashing or being interrupted),
    using the __del__ class method.

    """

    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def unlock(self, force=False):
        """
        Remove current lock.

        This function does not crash if it is unable to properly
        delete the lock file and directory. The reason is that it
        should be allowed for multiple jobs running in parallel to
        unlock the same directory at the same time (e.g. when reaching
        their timeout limit).

        """

        lock_file = os.path.join(self.tmp_dir, 'lock')
        if not force:
            try:
                with open(lock_file) as f:
                    owner = f.readlines()[0].strip()
                    pid, _, hname = owner.split('_')
                    if pid != str(os.getpid()) or hname != hostname:
                        return
            except Exception:
                pass

        try:
            os.remove(lock_file)
        except Exception:
            pass
        try:
            os.rmdir(self.tmp_dir)
        except Exception:
            pass
