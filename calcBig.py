import cPickle
import gzip
import time
import os
import sys
import cPickle as pickle
import gc
import numpy as np
from time import sleep


f1 = sys.argv[1]
f2 = sys.argv[2]

RESa0 = np.zeros((1000,1000))
RESb0 = np.zeros((1000,1000))
RESa = np.zeros((1000,1000))
RESb = np.zeros((1000,1000))

pos = 0
st = 1
dwa = 0
with open(f1) as ff1:   
    for line in ff1: 
        if st == 1:
            st = 0
            continue
        _, v = line.split(',')
        if dwa==0:
            RESa0[pos/1000][pos%1000] = float(v)
        else:
            RESb0[pos/1000][pos%1000] = float(v)
        pos+=1
        if pos == 1000000:
            pos = 0
            dwa = 1
print 'init'
# stala = 0.5
# Jmean = []
# for j in range(1000):
    # Jmean.append(np.mean(RESa0[j,:]**3))
# for i in range(1000):
    # opt = np.mean(RESa0[i,:]**3)
    # for j in range(1000):
        # RESa[i][j] = RESa0[i][j] /  (stala+(opt*Jmean[j])**0.5)  

# print 'a done'      
# Jmean = []
# for j in range(1000):
    # Jmean.append(np.mean(RESb0[j,:]**3))  
# for i in range(1000):
    # opt = np.mean(RESb0[i,:]**3)
    # for j in range(1000):
        # RESb[i][j] = RESb0[i][j] /  (stala+(opt*Jmean[j])**0.5)          
        
        
stala = 3.5
Jmean = []
for j in range(1000):
    Jmean.append(np.percentile(RESa0[j,:]**3, 98))
for i in range(1000):
    opt = np.percentile(RESa0[i,:]**3, 98)
    for j in range(1000):
        ss = RESa0[:,i]*RESa0[:,j] 
        z = np.percentile(ss, 98.5)  
        RESa[i][j] = (RESa0[i][j] * (10-z))  /  (stala+(opt*Jmean[j]**0.25) )

print 'a done'      
Jmean = []
for j in range(1000):
    Jmean.append(np.percentile(RESb0[j,:]**3, 98))  
for i in range(1000):
    opt = np.percentile(RESb0[i,:]**3, 98)
    for j in range(1000):
        ss = RESb0[:,i]*RESb0[:,j] 
        z = np.percentile(ss, 98.5)  
        RESb[i][j] = (RESb0[i][j] * (10-z))  /  (stala+(opt*Jmean[j]**0.25) )             
        
        
        
        
        
        
        
print 'b done'
RESa /= np.amax(RESa)
RESb /= np.amax(RESb)
print 'ending' 
f = open(f2, 'w')
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
print 'end.'        
f.close() 

