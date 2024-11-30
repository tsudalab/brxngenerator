from __future__ import absolute_import, print_function, division
from datetime import datetime
from six.moves import xrange

__authors__ = "Ian Goodfellow"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"


def disturb_mem():
    global l
    now = datetime.now()
    ms = now.microsecond
    ms = int(ms)
    n = ms % 1000
    m = ms // 1000
    l = [[0] * m for i in xrange(n)]
