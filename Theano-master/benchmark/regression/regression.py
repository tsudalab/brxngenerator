from __future__ import absolute_import, print_function, division
import theano
import numpy as N
from theano import tensor as T
from theano.tensor import nnet as NN
from six.moves import xrange
from theano.compile import module as M

class RegressionLayer(M.Module):
    def __init__(self, input = None, target = None, regularize = True):
        super(RegressionLayer, self).__init__() #boilerplate
        self.regularize = regularize
        if not input:
            input = T.matrix('input')
        if not target:
            target = T.matrix('target')
        self.stepsize = T.scalar()  # a stepsize for gradient descent
        self.w = T.matrix()  #the linear transform to apply to our input points
        self.b = T.vector()  #a vector of biases, which make our transform affine instead of linear
        self.activation = T.dot(input, self.w) + self.b
        self.prediction = self.build_prediction()
        self.classification_cost = self.build_classification_cost(target)
        self.regularization = self.build_regularization()
        self.cost = self.classification_cost
        if self.regularize:
            self.cost = self.cost + self.regularization
        self.grad_w, self.grad_b, grad_act = T.grad(self.cost, [self.w, self.b, self.prediction])
        print('grads', self.grad_w, self.grad_b)
        self.update = M.Method([input, target],
                               [self.cost, self.grad_w, self.grad_b, grad_act],
                               updates={self.w: self.w - self.stepsize * self.grad_w,
                                        self.b: self.b - self.stepsize * self.grad_b})
        self.apply = M.Method(input, self.prediction)
    def params(self):
        return self.w, self.b
    def _instance_initialize(self, obj, input_size = None, target_size = None,
                             seed = 1827, **init):
        if input_size and target_size:
            sz = (input_size, target_size)
            rng = N.random.RandomState(seed)
            obj.w = rng.uniform(size = sz, low = -0.5, high = 0.5)
            obj.b = N.zeros(target_size)
            obj.stepsize = 0.01
        M.default_initialize(obj,**init)
    def build_regularization(self):
        return T.zero() # no regularization!


class SpecifiedRegressionLayer(RegressionLayer):
    """ XE mean cross entropy"""
    def build_prediction(self):
        return NN.sigmoid(self.activation)
    def build_classification_cost(self, target):
        self.classification_cost_matrix = (target - self.prediction)**2
        self.classification_costs = T.sum(self.classification_cost_matrix, axis=1)
        return T.sum(self.classification_costs)
    def build_regularization(self):
        self.l2_coef = T.scalar() # we can add a hyper parameter if we need to
        return self.l2_coef * T.sum(self.w * self.w)


class PrintEverythingMode(theano.Mode):
    def __init__(self, linker, optimizer=None):                                                       
        def print_eval(i, node, fn): 
            print(i, node, [input[0] for input in fn.inputs], end=' ')                                         
            fn()
            print([output[0] for output in fn.outputs])
        wrap_linker = theano.gof.WrapLinkerMany([linker], [print_eval])
        super(PrintEverythingMode, self).__init__(wrap_linker, optimizer)                             


def test_module_advanced_example():

    profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
    profmode = PrintEverythingMode(theano.gof.OpWiseCLinker(), 'fast_run')

    data_x = N.random.randn(4, 10)
    data_y = [ [int(x)] for x in (N.random.randn(4) > 0)]


    model = SpecifiedRegressionLayer(regularize = False).make(input_size = 10,
                       target_size = 1,
                       stepsize = 0.1,
                       mode=profmode)

    for i in xrange(1000):
       xe, gw, gb, ga = model.update(data_x, data_y)
       if i % 100 == 0:
           print(i, xe)
           pass

    print("final weights:", model.w)
    print("final biases:", model.b)

    profmode.print_summary()
