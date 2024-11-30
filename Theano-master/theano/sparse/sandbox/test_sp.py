from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest
import sys
import time
import unittest

import theano.sparse
if not theano.sparse.enable_sparse:
    raise SkipTest('Optional package sparse disabled')

import scipy.sparse
from scipy.signal import convolve2d
import scipy.sparse as sparse
import numpy
from six.moves import xrange

from theano import function, tensor
import theano
from theano.compat import next
from theano.sparse.sandbox import sp
from theano.sparse.tests.test_basic import random_lil
from theano.tests import unittest_tools as utt
from theano.sparse import verify_grad_sparse
from theano.sparse.tests.test_basic import sparse_random_inputs
from theano.tests.unittest_tools import attr


class TestSP(unittest.TestCase):
    @attr('slow')
    def test_convolution(self):

        bsize = 10     # batch size
        imshp = (28, 28)
        kshp = (5, 5)
        nkern = 5
        ssizes = ((1, 1), (2, 2), (3, 3), (4, 4))
        convmodes = ('full', 'valid')

        bias = tensor.dvector()
        kerns = tensor.dmatrix()
        input = tensor.dmatrix()
        rng = numpy.random.RandomState(3423489)
        filters = rng.randn(nkern, numpy.prod(kshp))
        biasvals = rng.randn(nkern)

        for mode in ('FAST_COMPILE', 'FAST_RUN'):  # , profmode):
            ttot, ntot = 0, 0
            for conv_mode in convmodes:
                for ss in ssizes:

                    output, outshp = sp.convolve(kerns, kshp, nkern, input,\
                            imshp, ss, bias=bias, mode=conv_mode)
                    f = function([kerns, bias, input], output, mode=mode)

                    img2d = numpy.arange(bsize * numpy.prod(imshp)).reshape(( \
                                                            bsize,) + imshp)
                    img1d = img2d.reshape(bsize, -1)

                    filtersflipped = numpy.zeros((nkern,) + kshp)
                    for k in range(nkern):
                        it = reversed(filters[k, :])
                        for i in range(kshp[0]):
                            for j in range(kshp[1]):
                                filtersflipped[k, i, j] = next(it)

                    if conv_mode == 'valid':
                        fulloutshp = numpy.array(imshp) - numpy.array(kshp) + 1
                    else:
                        fulloutshp = numpy.array(imshp) + numpy.array(kshp) - 1
                    ntime1 = time.time()
                    refout = numpy.zeros((bsize,)+tuple(fulloutshp)+(nkern,))
                    for b in range(bsize):
                        for n in range(nkern):
                            refout[b, ..., n] = convolve2d(img2d[b, :, :],
                                                         filtersflipped[n, ...],
                                                         conv_mode)
                    ntot += time.time() - ntime1

                    bench1 = refout[:, 0::ss[0], 0::ss[1], :].reshape(bsize, -1, nkern)
                    bench1 += biasvals.reshape(1, 1, nkern)

                    bench1 = numpy.swapaxes(bench1, 1, 2)
                    ttime1 = time.time()
                    out1 = f(filters, biasvals, img1d)
                    ttot += time.time() - ttime1
                    temp = bench1.flatten() - out1.flatten()

                    assert (temp < 1e-5).all()







    def test_multilayer_conv(self):
        bsize = 10     # batch size
        imshp = (5, 5)
        kshp = ((3, 3), (2, 2))
        nkerns = (3, 6)  # per output pixel
        ssizes = (((1, 1), (2, 2)),)
        convmodes = ('full',)  # 'valid',)

        kerns = [tensor.dmatrix(), tensor.dmatrix()]
        input = tensor.dmatrix()
        rng = numpy.random.RandomState(3423489)

        img2d = numpy.arange(bsize*numpy.prod(imshp)).reshape((bsize,)+imshp)
        img1d = img2d.reshape(bsize, -1)

        for mode in ('FAST_COMPILE', 'FAST_RUN'):
            for conv_mode in convmodes:
                for ss in ssizes:

                    l1hid, l1shp = sp.convolve(kerns[0], kshp[0],\
                            nkerns[0], input, imshp, ss[0], mode=conv_mode)
                    l1propup = function([kerns[0], input], l1hid, mode=mode)

                    l1kernvals = numpy.arange(nkerns[0]*numpy.prod(kshp[0])).reshape(nkerns[0], numpy.prod(kshp[0]))
                    l1hidval = l1propup(l1kernvals, img1d)

                    l2hid, l2shp = sp.convolve(kerns[1], kshp[1],\
                            nkerns[1], l1hid, l1shp, ss[1], mode=conv_mode)
                    l2propup = function([kerns[1], l1hid], l2hid, mode=mode)

                    l2kernvals = numpy.arange(nkerns[1]*numpy.prod(kshp[1])*nkerns[0]).reshape(nkerns[1], numpy.prod(kshp[1])*nkerns[0])
                    l1hidval = numpy.arange(numpy.size(l1hidval)).reshape(l1hidval.shape)

                    l2hidval = l2propup(l2kernvals, l1hidval)

    def test_maxpool(self):
        maxpoolshps = ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6))
        imval = numpy.random.rand(4, 5, 10, 10)

        images = tensor.dmatrix()
        for maxpoolshp in maxpoolshps:

            output, outshp = sp.max_pool(images, imval.shape[1:], maxpoolshp)
            f = function([images, ], [output, ])
            output_val = f(imval.reshape(imval.shape[0], -1))

            my_output_val = numpy.zeros((imval.shape[0], imval.shape[1],
                                     imval.shape[2] // maxpoolshp[0],
                                     imval.shape[3] // maxpoolshp[1]))
            assert numpy.prod(my_output_val.shape[1:]) == numpy.prod(numpy.r_[imval.shape[1], outshp])

            for n in range(imval.shape[0]):
                for k in range(imval.shape[1]):
                    for i in range(imval.shape[2] // maxpoolshp[0]):
                        for j in range(imval.shape[3] // maxpoolshp[1]):
                            ii, jj = i*maxpoolshp[0], j*maxpoolshp[1]
                            patch = imval[n, k, ii:ii+maxpoolshp[0], jj:jj+maxpoolshp[1]]
                            my_output_val[n, k, i, j] = numpy.max(patch)
            my_output_val = my_output_val.reshape(imval.shape[0], -1)
            assert numpy.all(output_val == my_output_val)

            def mp(input):
                output, outshp = sp.max_pool(input, imval.shape[1:], maxpoolshp)
                return output
            utt.verify_grad(mp, [imval.reshape(imval.shape[0], -1)])


if __name__ == '__main__':
    if 0:
        test_remove0()
        exit()
    if 1:
        testcase =  TestSP
        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        unittest.main()
