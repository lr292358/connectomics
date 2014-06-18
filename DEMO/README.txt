Lukasz Romaszko (Lukasz 8000). 
Demo.
2 May 2014. Version 1.

Libraries versions:
Python 2.6.6 with Numpy 1.4.1 [Linux] (Demo tested also on Python 2.7.6 & Numpy 1.8.0 & Windows)
Theano 0.6.0 (the newest)

In general, the code will probably run on any Python 2.x with Numpy & newest Theano.
The Demo can be run even on a CPU in 15 minutes. It tests whole code on small networks.


It uses small network no.5 (ClusteringCoefficent05) to learn (only 1 epoch), 
treats small no.4 and small no.6 as validation and test networks, and produces the output
like this which is send to Kaggle (here, of 2 x 10 000 predictions) to the file 'res_small_4_6.csv'. 
The AUC scores are also computed and printed to the output.

Results of the run are saved in out.txt. 
Network 4 gets 89.91 % AUC, 
network 6: 95.46 % AUC.

running: 
1. 
put the directory 'data/small' (containing small networks data) into the extracted directory.
You can also instead set the path to the 'small' directory in the 'run.py' file, line 28:
"""
#path to data containing 'small' directory
path = "data"
"""

2. Run:
$ python run.py

The normal predictions were executed on GPUs nVidia tesla K20. 
Normal predictions to submit on Kaggle will require execution on GPUs.
In general, decreasing training set size allows faster execution but lower accuracy.
Demo uses very small dataset and training is only run through one epoch.



