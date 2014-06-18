import cPickle
import gzip
import time
import os
import sys
import cPickle as pickle
import gc
import numpy as np
from time import sleep
import auc
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
import theano.printing
from collections import OrderedDict

from logisticRegression import LogisticRegression
from layers import DropoutHiddenLayer, HiddenLayer2d, HiddenLayer, ConvolutionalHiddenSoftmax, ConvolutionalLayer
import warnings
warnings.filterwarnings('ignore')


L = 330
n_epochs = 20
Q = 14 
NUM_TRAIN = 1200000  #(use multiplicity of 50'000)
MINIREAD = 1
batch_size= 1000

############
############ uncomment these lines below to verify if the code runs correctly, execution around 15 times faster
############
# n_epochs = 4
# Q = 4
# NUM_TRAIN = 250000 
# MINIREAD = 4
# batch_size= 1000


THREAD = 20
span = 1
NN = 1000   
POOL = 10

Knormal =  1794 * 100 / MINIREAD
learning_rate0 = 0.2;


def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)

def read(s, sn ,sp, Kile):
    P=[]; lenn = []; nott = 0
    _nps = []
    _s = []
    with open(path+sp) as ff:   
        for line in ff: 
            x,y = line.split(',')
            P.append([float(x),float(y)])      
    print "opening"    
    with open(path+s) as f:
        rlast = []; cnt = 0; arrayprev = []; Ti = []; ile = 0
        for line in f:
            if cnt % 17940 == 0:
                print str(cnt/1794), "% ",
            if cnt != 0:  
                pos = 0; r = []; rr2 = np.zeros(NN); rr = np.zeros(NN); rp = []
                for x in line.split(','):
                    x_f = float(x)
                    rp.append(x_f)
                for x in rp:
                    val2 = x - arrayprev[pos] 
                    rr[pos] = val2 # to sum
                    pos+=1  
                nps = np.sum(rr)
                _w = [_x for _x in rr if _x >= 0.2]
                _wn = len(_w)
                if nps <  THREAD :
                    if nott > 0:
                        lenn.append(nott); 
                        ile+=nott
                        nott = 0
                    else:
                        nott -=1
                else: 
                    if nott <= 0:
                        nott = 1  
                    else:
                        nott += 1
                    pos+=1                  
                if nott >= 1:  
                    Ti.append(rr) 
                    _nps.append(nps)
                    if nott==1:
                        _s.append(1)
                    else:
                        _s.append(1)
                arrayprev = rp
            else:
                arrayprev = [float(x) for x in line.split(',')]
                
            if cnt >  Kile + 10:
                break
            cnt+=1  
    C = [[0]*len(rr)]*len(rr)
    C = np.asarray(C)
    print "\n\n              selected frames number = ", ile, "\n\n"
    if sn != None:
        with open(path+sn) as ff:   
            for line in ff: 
                a,b,w = line.split(',')
                a = int(a); b = int(b); w = int(w)
                if w==1:
                    C[a-1][b-1] = 1;
    print "trans..."
    Tprim = np.empty((len(rr)+2, ile), np.float32) ##############
    for j in range(len(rr)):
        a = []
        for i in range(ile):
            Tprim[j][i] = Ti[i][j]
    for i in range(ile):
        Tprim[1000][i] = _nps[i]
    for i in range(ile):
        Tprim[1001][i] = _s[i]
    gc.collect()
    print "AVG SPLIT LEN: ", np.mean(lenn)
    return Tprim, C, P

        

def learnAndPredict(Ti, C, TOList):
 
    rng = np.random.RandomState(SEED)
    learning_rate = learning_rate0
    print np.mean(Ti[1000,:])
    aminW = np.amin(Ti[:1000,:])
    amaxW = np.amax(Ti[:1000,:]) 
    Ti[:1000,:] = (Ti[:1000,:] - aminW) / (amaxW - aminW)
    astdW = np.std(Ti[:1000,:])
    ameanW = np.mean(Ti[:1000,:])
    Ti[:1000,:] = (Ti[:1000,:] - ameanW) / astdW
    aminacW = np.amin(Ti[1000,:])
    amaxacW = np.amax(Ti[1000,:])
    print aminW, amaxW, aminacW, amaxacW
    Ti[1000,:] =  (Ti[1000,:] - aminacW) / (amaxacW - aminacW)
    astdacW = np.std(Ti[1000,:])
    ameanacW = np.mean(Ti[1000,:])
    Ti[1000,:] =  (Ti[1000,:] - ameanacW) / astdacW
    
    ile__ = len(TOList)
    ileList = np.zeros(ile__)
    for titer in range(len(TOList)):
        print np.mean(TOList[titer][1000,:])
        TOList[titer][:1000,:] = (TOList[titer][:1000,:] - aminW)/(amaxW - aminW)
        TOList[titer][:1000,:] = (TOList[titer][:1000,:] - ameanW)/astdW
        TOList[titer][1000,:] =  (TOList[titer][1000,:] - aminacW)/(amaxacW - aminacW)
        TOList[titer][1000,:] =  (TOList[titer][1000,:] - ameanacW)/astdacW
        _, ileList[titer] = TOList[titer].shape
        
    _, ile = Ti.shape
    N = NN
  
    data = []; yyy = []; need = 1; BYL = {}; j= 0; dwa = 0; ONES = []; ZEROS = []
    for i in range(NN):
        for j in range(NN):
            if i!= j:
                if C[i][j]==1:
                    ONES.append((i,j))
                else:
                    ZEROS.append((i,j))
    Nones = len(ONES)
    rng.shuffle(ONES)
    Nzeros = len(ZEROS)
    print Nones
    print Nzeros
    Needed = NUM_TRAIN/2
    onesPerPair = Needed / Nones + 1
    onesIter = 0
    jj = 0
    while jj < NUM_TRAIN:
        if jj%300000 == 0:
            print jj/300000,
        need = 1 - need
        if need == 1:
            pairNo = onesIter % Nones
            ppp = onesIter / Nones
            s,t = ONES[pairNo]
            shift = rng.randint(0, ile - L)
            onesIter += 1
        if need == 0:
            zer = rng.randint(Nzeros)
            s,t = ZEROS[zer]
            del ZEROS[zer]
            Nzeros -= 1
            shift = rng.randint(0, ile - L)
        x = np.hstack(( Ti[s][shift:shift+L], Ti[t][shift:shift+L], Ti[1000][shift:shift+L]))
        y = C[s][t]
        data.append(x); yyy.append(y)
        jj+=1

    data = np.array(data, dtype=theano.config.floatX)  
    is_train = np.array(  ([0]*96 + [1,1,2,2]) * (NUM_TRAIN / 100))
    yyy = np.array(yyy)
    
    train_set_x0, train_set_y0 = np.array(data[is_train==0]), yyy[is_train==0]
    test_set_x,   test_set_y = np.array(data[is_train==1]), yyy[is_train==1]
    valid_set_x, valid_set_y = np.array(data[is_train==2]), yyy[is_train==2]
    n_train_batches = len(train_set_y0) / batch_size
    n_valid_batches = len(valid_set_y)  / batch_size
    n_test_batches  = len(test_set_y)  / batch_size  
    epoch = T.scalar() 
    index = T.lscalar() 
    x = T.matrix('x')   
    inone2 = T.matrix('inone2') 
    y = T.ivector('y') 
    print '... building the model'
#-------- my layers -------------------
    
    #---------------------
    layer0_input = x.reshape((batch_size, 1, 3, L))
    Cx = 5
    layer0 = ConvolutionalLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 3, L),
            filter_shape=(nkerns[0], 1, 2, Cx), poolsize=(1, 1), fac = 0)
    ONE = (3 - 2 + 1) / 1
    L2 = (L - Cx + 1) / 1
    #---------------------
    Cx2 = 5
    layer1 = ConvolutionalLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], ONE, L2),
            filter_shape=(nkerns[1], nkerns[0], 2, Cx2), poolsize=(1, 1), activation=ReLU, fac = 0)
    ONE = (ONE - 2 + 1) /1
    L3 = (L2 - Cx2 + 1) /1
    #---------------------
    Cx3 = 1
    layer1b = ConvolutionalLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], ONE, L3),
            filter_shape=(nkerns[2], nkerns[1], 1, Cx3), poolsize=(1, POOL), activation=ReLU, fac = 0)
    ONE = (ONE - 1 + 1) /1
    L4 = (L3 - Cx3 + 1) /POOL
    
    REGx = 100
    #---------------------    
    layer2_input = layer1b.output.flatten(2) 
    print layer2_input.shape
    use_b = False
    layer2 =         HiddenLayer(rng, input=layer2_input, n_in=nkerns[2]*L4 , n_out=REGx, activation=T.tanh,
                                 use_bias = use_b)
    layer3 =  LogisticRegression(input=layer2.output, n_in=REGx, n_out=2)
 
    
    cost = layer3.negative_log_likelihood(y)
    out_x2 = theano.shared(np.asarray(np.zeros((N,L)), dtype=theano.config.floatX))
    inone2 = theano.shared(np.asarray(np.zeros((1,L)), dtype=theano.config.floatX))
    inone3 = theano.shared(np.asarray(np.zeros((1,L)), dtype=theano.config.floatX))
    inone4 = theano.shared(np.asarray(np.zeros((1,L)), dtype=theano.config.floatX))
    test_set_x = theano.shared(np.asarray(test_set_x, dtype=theano.config.floatX))
    train_set_x = theano.shared(np.asarray(train_set_x0, dtype=theano.config.floatX))
    train_set_y = T.cast(theano.shared(np.asarray(train_set_y0, dtype=theano.config.floatX)), 'int32')
    test_set_y = T.cast(theano.shared(np.asarray(test_set_y, dtype=theano.config.floatX)), 'int32')
    valid_set_y =  T.cast(theano.shared(np.asarray(valid_set_y, dtype=theano.config.floatX)), 'int32')
    valid_set_x = theano.shared(np.asarray(valid_set_x, dtype=theano.config.floatX))   
    
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

       
    mom_start = 0.5; mom_end = 0.98;  mom_epoch_interval = n_epochs * 1.0
    #### @@@@@@@@@@@
    class_params0  =  [layer3, layer2, layer1, layer1b, layer0]  
    class_params = [ param for layer in class_params0 for param in layer.params ]

    gparams = []
    for param in class_params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    gparams_mom = []
    for param in class_params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam
    for param, gparam_mom in zip(class_params, gparams_mom):
        stepped_param = param + updates[gparam_mom]
        squared_filter_length_limit = 15.0
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param

    output = cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    
    keep = theano.function([index], layer3.errorsFull(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]}, on_unused_input='warn')

    timer = time.clock()
    print "finished reading", (timer - start_time0) /60. , "minutes "
             
    # TRAIN MODEL # 
    print '... training'
    validation_frequency = n_train_batches; best_params = None; best_validation_loss = np.inf
    best_iter = 0; test_score = 0.;  epochc = 0;
    
    while (epochc < n_epochs):
        epochc = epochc + 1            
        learning_rate = learning_rate0 * (1.2 - ((1.0 * epochc)/n_epochs))
        for minibatch_index in xrange(n_train_batches):      
            iter = (epochc - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(epochc, minibatch_index)  
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(' %i) err %.2f ' %  (epochc, this_validation_loss/10)), L, nkerns, REGx, "|", Cx, Cx2, Cx3, batch_size
                if this_validation_loss < best_validation_loss or epochc % 30 == 0:
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') % (epochc, minibatch_index + 1, n_train_batches, test_score/10))
    ############        
    timel = time.clock()
    print "finished learning", (timel - timer) /60. , "minutes "
    ppm = theano.function([index], layer3.pred_proba_mine(),
        givens={
            x: T.horizontal_stack(T.tile(inone2, (batch_size ,1)), 
               out_x2[index * batch_size: (index + 1) * batch_size], T.tile(inone3, (batch_size ,1))),
            y: train_set_y[0 * (batch_size): (0 + 1) * (batch_size)]
            }, on_unused_input='warn')

    NONZERO = (N*N-N)
    gc.collect()
    RESList = [np.zeros((N,N)) for it in range(ile__)]
    for __net in range(ile__):
        TO = TOList[__net]
        ileO = ileList[__net]
        RES  = RESList[__net]
        shift = 0.1 
        DELTAshift = (ileO-L) / (Q-1)
        print "DELTAshift:", DELTAshift
        for q in range (Q):
            dataO = [];  print (q+1),"/", Q , "  ",
            out_x2.set_value(np.asarray(np.array(TO[:,shift:shift+L]), dtype=theano.config.floatX)) 
            PARTIAL = np.zeros((N,N))
            inone3.set_value(np.asarray(np.array(TO[1000][shift:shift+L]).reshape(1,L), dtype=theano.config.floatX))
            for i in range(N):
                inone2.set_value(np.asarray(np.array(TO[i][shift:shift+L]).reshape(1,L), dtype=theano.config.floatX))
                p = [ppm(ii) for ii in xrange( N / batch_size)]
                for pos in range(N):
                    if pos != i:
                        PARTIAL[i][pos] += p[pos / batch_size][pos % batch_size][1]
            for i in range(N):
                for j in range(N):
                    RES[i][j] += PARTIAL[i][j]
            shift += DELTAshift
        print "Finished", __net
        RESList[__net] = RES/np.max(RES)            
        gc.collect()
        
    end_time = time.clock()
    print "finished predicting", (end_time - timel) /60. , "minutes ", str(nkerns), "using SEED = ", SEED
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time0) / 60.))
    return RESList


    
if __name__ == '__main__':
    MY = 9 #(GPU)
    VER = 1
    if len(sys.argv)>1:  # select random sequence seed
        VER = int(sys.argv[1])    
   
    nkerns = [18, 40, 15]
    if VER == 1:
        SEED = 8001
    if VER == 2:
        SEED = 80001
    if VER == 3: 
        SEED = 888
    if VER == 4: 
        SEED = 8881
    if VER == 5: 
        SEED = 8001 
    if VER == 6: 
        SEED = 100
    if VER == 7: 
        SEED = 18000
    if VER == 8: 
        SEED = 80801
    
    
    start_time0 = time.clock()
    print THREAD
    if MY == 1 or MY == 5 or MY == 2 or MY==9:    
        path = "/firstly/set/path/here"
        name = "normal-1"
        s = "/"+name+"/fluorescence_"+name+".txt"
        sn = "/"+name+"/network_"+name+".txt"
        sp = "/"+name+"/networkPositions_"+name+".txt"
        print name
        TN1, CN1 , PN1 = read(s,sn,sp, Knormal)
        gc.collect()
        name0 = name
        name = "normal-3"
        s = "/"+name+"/fluorescence_"+name+".txt"
        sn = "/"+name+"/network_"+name+".txt"
        sp = "/"+name+"/networkPositions_"+name+".txt"
        print name
        TN2, CN2, PN2 = read(s,sn,sp, Knormal)
        gc.collect()
        name = "normal-2"
        s = "/"+name+"/fluorescence_"+name+".txt"
        sn = "/"+name+"/network_"+name+".txt"
        sp = "/"+name+"/networkPositions_"+name+".txt"
        print name
        TN3, CN3 , PN3 = read(s,sn,sp, Knormal)
        gc.collect()
        if MY == 9:         
            print "reading valid..."
            s = "/valid/fluorescence_valid.txt"
            sn = None
            sp = "/valid/networkPositions_valid.txt"
            TV, _, PV = read(s,sn,sp, Knormal)
            print "reading test..."
            s = "/test/fluorescence_test.txt"
            sn = None
            sp = "/test/networkPositions_test.txt"
            TT, _, PT = read(s,sn,sp, Knormal)

            [RN2, RT, RV, RN3] = learnAndPredict(TN1, CN1, [TN2, TT, TV, TN3])
            suff = np.random.randint(10000)
            f = open("./res_ver"+str(VER)+".csv", 'w')
            f.write("NET_neuronI_neuronJ,Strength\n")
            for i in range (1000):
                for j in range (1000):
                    f.write("valid_" +str(i+1)+"_"+str(j+1)+","+str(RV[i][j])+"\n")
            for i in range (1000):
                for j in range (1000):
                    f.write("test_" +str(i+1)+"_"+str(j+1)+","+str(RT[i][j])+"\n")
            f.close()
            print "Wrote solution of VER ==", str(VER)        
            RN2_ = RN2.flatten().tolist()
            a = auc.auc(CN2.flatten().tolist(),RN2_)
            RN3_ = RN3.flatten().tolist()
            a2 = auc.auc(CN3.flatten().tolist(),RN3_)
            print ("RES: %.2f learning (%.2f, %.2f)" % ((a+a2)*50, a*100, a2*100 ))


