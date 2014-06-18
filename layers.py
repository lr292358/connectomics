import numpy as np
import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import theano.printing
import theano.tensor.shared_randomstreams

from logisticRegression import LogisticRegression

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
            
class HiddenLayer2d(object):
    def __init__(self, rng, input, n_in, n_in2, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_out, n_in, 1, n_in2)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W,) + self.b
        else:
            lin_output = T.tensordot(input, self.W, axes = [[1,2,3],[1,2,3]])

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]            
            
            


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

        

class ConvolutionalLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(4, 4), activation=T.tanh, fac = 0, W=None, b=None):
        """
        Allocate a ConvolutionalLayer with shared variable internal parameters.
        :type rng: numpy.random.RandomState        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,   filter height,filter width)
        :type image_shape: tuple or list of length 4        :param image_shape: (batch size, num input feature maps,  image height, image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape    
        # there are "num input feature maps * filter height * filter width"        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        # initialize weights with random weights
        W_bound = 1.5 * np.sqrt(6. / (fan_in + fan_out))
        initt =rng.uniform(low=-W_bound, high=W_bound, size=filter_shape)
        if fac == 1:
            mask = rng.binomial(n=1, p= 1 - 0.2, size=filter_shape)           
            initt = initt * mask
            self.mm = np.asarray(mask,  dtype=theano.config.floatX)
        self.W = W
        if W is None:
            self.W = theano.shared(np.asarray(initt,  dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = b
        if b is None:
            self.b = theano.shared(value=b_values, borrow=True)
        # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W,
        filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=self.conv_out, ds=poolsize, ignore_border=True)
        
        # add the bias term. Since the bias is a vector (1D array), we first        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map        # width & height
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        
class ConvolutionalHiddenSoftmax(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(4, 4), activation=T.tanh, fac = 0, W=None, b=None, WSoft = None, bSoft = None):
  
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape    
        # there are "num input feature maps * filter height * filter width"        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        # initialize weights with random weights
        W_bound = 1.5 * np.sqrt(6. / (fan_in + fan_out))
        initt =rng.uniform(low=-W_bound, high=W_bound, size=filter_shape)
        if fac == 1:
            mask = rng.binomial(n=1, p= 1 - 0.2, size=filter_shape)           
            initt = initt * mask
            self.mm = np.asarray(mask,  dtype=theano.config.floatX)
        self.W = W
        if W is None:
            self.W = theano.shared(np.asarray(initt,  dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = b
        if b is None:
            self.b = theano.shared(value=b_values, borrow=True)
        # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=self.conv_out, ds=poolsize, ignore_border=True)
        self.WSoft=WSoft
        self.bSoft=bSoft
        # add the bias term. Since the bias is a vector (1D array), we first        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map        # width & height
       # T.nnet.softmax(T.dot(input, self.WSoft) + self.bSoft)
        self.outputH = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(3)
        

        #self.output, updates = theano.map(lambda x: T.nnet.softmax(T.tensordot(x, self.WSoft)  + self.bSoft), self.outputH.dimshuffle(0,2,1))
        
        o, updates = theano.scan(fn = lambda x: T.nnet.softmax(T.dot(x, WSoft)  + bSoft),
                                              outputs_info=None,
                                              sequences=[self.outputH.dimshuffle(0,2,1)],
                                              )
       
        self.output = o        