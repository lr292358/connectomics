import cPickle
import gzip
import time
import os
import sys
import cPickle as pickle
import gc
import numpy as np
from time import sleep


fout = sys.argv[1]
N = len(sys.argv)-2

RES = np.zeros((N, 1000*1000*2))
for i in range(N):
    pos = 0
    st = 1
    f1 = sys.argv[i+2]
    
    with open(f1) as ff1:   
        for line in ff1: 
            if st == 1:
                st = 0
                continue
            _, v = line.split(',')
            RES[i][pos] = float(v)
            pos+=1
    print 'done', f1    
    ff1.close() 


f = open(fout, 'w')
f.write("NET_neuronI_neuronJ,Strength\n")
pos = 0
for i in range (1000):
    for j in range (1000):
        ile = 0
        for ii in range(N):
            ile += RES[ii][pos]
        ile /= N
        f.write("valid_" +str(i+1)+"_"+str(j+1)+","+str(ile)+"\n")
        pos+=1
for i in range (1000):
    for j in range (1000):
        ile = 0
        for ii in range(N):
            ile += RES[ii][pos]
        ile /= N
        f.write("test_" +str(i+1)+"_"+str(j+1)+","+str(ile)+"\n")
        pos+=1
f.close() 

