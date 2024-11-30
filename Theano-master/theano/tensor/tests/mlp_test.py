"""
This is a minimized version of the mlp.py in the tutorial. We removed stuff that make this mlp don't work.
But this test a bug that we saw. This bug made the Shape_i object not being lifted, that caused the CrossentropySoftmax... op not being inserted.
"""
from __future__ import absolute_import, print_function, division

__docformat__ = 'restructedtext en'


import numpy

import theano
import theano.tensor as T
from theano.compat import OrderedDict


def gen_data():

    train_set = (numpy.asarray(numpy.random.rand(10000, 784), dtype='float32'),
               numpy.asarray(numpy.random.rand(10000)*10, dtype='int64'))
    valid_set = (numpy.asarray(numpy.random.rand(10000, 784), dtype='float32'),
               numpy.asarray(numpy.random.rand(10000)*10, dtype='int64'))
    test_set = (numpy.asarray(numpy.random.rand(10000, 784), dtype='float32'),
               numpy.asarray(numpy.random.rand(10000)*10, dtype='int64'))
    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, name_prefix=''):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                                name=name_prefix+'W')

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return T.log(self.p_y_given_x[T.arange(y.shape[0]), y])


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh, name_prefix=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        self.input = input

        W_values = numpy.asarray( rng.uniform( \
              low=-numpy.sqrt(6./(n_in+n_out)), \
              high=numpy.sqrt(6./(n_in+n_out)), \
              size=(n_in, n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name=name_prefix+'W')

        self.output = T.dot(input, self.W)
        self.params = [self.W]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermidiate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                 n_in=n_in, n_out=n_hidden,
                                 activation=T.tanh, name_prefix='hid_')

        self.logRegressionLayer = LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out, name_prefix='log_')

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def test_mlp():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                         http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = gen_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    batch_size = 100    # size of the minibatch

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size


    index = T.lscalar()    # index to a [mini]batch
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y')  # the labels are presented as 1D vector of

    rng = numpy.random.RandomState(1234)

    classifier = MLP( rng=rng, input=x, n_in=28*28, n_hidden=500, n_out=10)

    cost = classifier.negative_log_likelihood(y).mean()

    gparams = []
    for param in classifier.params:
        gparam  = T.grad(cost, param)
        gparams.append(gparam)

    mode = theano.compile.get_default_mode().including('fast_run')

    updates2 = OrderedDict()

    updates2[classifier.hiddenLayer.params[0]] = T.grad(cost, classifier.hiddenLayer.params[0])
    train_model = theano.function( inputs=[index],
            updates=updates2,
            givens={
                x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]},
            mode=mode)
    assert any([isinstance(i.op, T.nnet.CrossentropySoftmax1HotWithBiasDx) for i in train_model.maker.fgraph.toposort()])

    train_model = theano.function( inputs=[index],
            updates=updates2,
            mode=mode.excluding('ShapeOpt'),
            givens={
                x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]})
    assert any([isinstance(i.op, T.nnet.CrossentropySoftmax1HotWithBiasDx) for i in train_model.maker.fgraph.toposort()])

if __name__ == '__main__':
    test_mlp()
