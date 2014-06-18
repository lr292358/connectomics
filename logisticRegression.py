import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        self.inp = input
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
    def errorsFull(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_pred, y)
        else:
            raise NotImplementedError()            
            
    def pred_proba_mine(self):
        pp = T.nnet.softmax(T.dot(self.inp, self.W) + self.b)
        return pp

    def getSVM(self):
        iii = self.inp
        return iii
        

class LogisticRegression2d(object):
    def __init__(self, input, nk, n_in, n_out, W=None, b=None):
        self.inp = input
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # o, updates = theano.scan(fn = lambda x: T.nnet.softmax(T.dot(x, WSoft)  + bSoft),
                                              # outputs_info=None,
                                              # sequences=[self.outputH],
                                              # )
       
        # self.output = o   
        print input.shape
     #   in2 = T.tensordot(input.dimshuffle(0,2,1), T.tile(self.W, (19, 1)), axes=[[1,2],[0,1]])  + T.tile(self.b, [19])
        
        o  =[ T.nnet.softmax(T.dot(input[z][i], self.W) + self.b) for i in range(19) for z in range(1000)]
        # o, updates = theano.scan(fn = lambda x: T.nnet.softmax(T.dot(x.T, self.W)  + self.b),
                                               # outputs_info=None,
                                               # sequences= [ in2 ] ,
                                               # )
        #o = T.nnet.softmax(T.tensordot(input.dimshuffle(0,2,1), T.tile(self.W, (19, 1)), axes=[[1,2],[0,1]])  + T.tile(self.b, [19]))
        self.p_y_given_x = T.mean(o, axis = 2)
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
    def errorsFull(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_pred, y)
        else:
            raise NotImplementedError()            
            
    def pred_proba_mine(self):
        pp = T.nnet.softmax(T.dot(self.inp, self.W) + self.b)
        return pp

    def getSVM(self):
        iii = self.inp
        return iii        