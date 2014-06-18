import cPickle
import gzip
import time
import os
import sys
import cPickle as pickle
import gc
import numpy as np
from time import sleep

#read
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

    
print 'merging...'

RESa0 = np.zeros((1000,1000))
RESb0 = np.zeros((1000,1000))
RESa = np.zeros((1000,1000))
RESb = np.zeros((1000,1000))

M=1000000
for i in range (1000):
    for j in range (1000):
        for k in range(N):
            RESa0[i,j] += RES[k][i*1000+j]
            RESb0[i,j] += RES[k][M+i*1000+j]
RESa0 /= N
RESb0 /= N

#normalize
print 'init normalization'

stala = 3.5
NUM = 98
NUM2 = 98.5
Jmean = []
for j in range(1000):
    Jmean.append(np.percentile(RESa0[j,:]**3, NUM)**0.25)
for i in range(1000):
    opt = np.percentile(RESa0[i,:]**3, NUM)
    for j in range(1000):
        ss = RESa0[:,i]*RESa0[:,j] 
        z = np.percentile(ss, NUM2)  
        RESa[i][j] = (RESa0[i][j] * (10 - z))  /  (stala+(opt*Jmean[j]) )

print 'valid done'      
Jmean = []
for j in range(1000):
    Jmean.append(np.percentile(RESb0[j,:]**3, NUM)**0.25)  
for i in range(1000):
    opt = np.percentile(RESb0[i,:]**3, NUM)
    for j in range(1000):
        ss = RESb0[:,i]*RESb0[:,j] 
        z = np.percentile(ss, NUM2)  
        RESb[i][j] = (RESb0[i][j] * (10 - z))  /  (stala+(opt*Jmean[j]) )             
        
#save        
print 'test done'
RESa /= np.amax(RESa)
RESb /= np.amax(RESb)
print 'ending...' 
f = open(fout, 'w')
f.write("NET_neuronI_neuronJ,Strength\n")
pos = 0
for i in range (1000):
    for j in range (1000):
        f.write("valid_" +str(i+1)+"_"+str(j+1)+","+str(RESa[i,j])+"\n")
        pos+=1
print 'ending.....'        
for i in range (1000):
    for j in range (1000):
        f.write("test_" +str(i+1)+"_"+str(j+1)+","+str(RESb[i,j])+"\n")
        pos+=1       
f.close() 
print 'end. Saved solution to', fout 


