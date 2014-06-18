Lukasz Romaszko (Lukasz 8000). 4 May 2014. Version 1.
This pack contains a Demo and the Solution which predict submission sent to Kaggle.
This file contains description how to run the Solution.

    The Solution is convolutional neural network applied to detect patterns in activity recordings of brain cells.
    Libraries versions:
    Python 2.6.6 with Numpy 1.4.1 [Linux] (Demo tested also on Python 2.7.6 & Numpy 1.8.0 & Windows)
    Theano 0.6.0 (the newest)

    In general, the code will probably run on any newer Python 2.x with Numpy & Theano.
    The Demo can be run even on a CPU in 15 minutes. It tests whole code on small networks.

    
    
_______________________
Demo:
    Directory /DEMO
    It uses small network no.5 (ClusteringCoefficent05) to learn (only 1 epoch), 
    treats small no.4 and small no.6 as validation and test networks, and produces the output
    like this which is send to Kaggle (here, of 2 x 10 000 predictions) to the file 'res_small_4_6.csv'. 
    The AUC scores are also computed and printed to the output.

    Results of the run are saved in out.txt. 
    Network 4: 89.91 % AUC, 
    Network 6: 95.46 % AUC.

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

        In general, decreasing training set size allows faster execution but lower accuracy.
        Demo uses very small dataset and training is only run through one epoch.
        
        
        
__________________________
Kaggle submission:


    The Solution uses only one GPU for a computation, the results were computed on nVidia Tesla K20 GPU on Linux (CentOS). 
    Training data size (NUM_TRAIN = 1.2 million examples) was set that solution fits in 5GB of card memory. 
    Execution time of a single run on K20 is around 15 hours. 
    The code should run on a fast GPUs with >=5GB memory. For instance, code was tested also on nVida GTX Titan.
    The submission was an average of 8 runs with different seeds. 
    The fastest would be computing all 8 predictions in the same time. This means using 8 GPUs in the same time, however
    the runs are completely not related to each other.
    
    In general, decreasing training set size allows faster execution but lower accuracy.
    The submission was created by computing 8 single predictions with different seeds and merging them with one script.
        (You can also firstly verify this whole code by uncommenting lines 35-39 (15 times faster execution on K20).)

    Running:
    1.  Set the path to the data directory in the line #411. 
        Required: Normal{1,2,3}, Validation, Test.

    2.  Run (standard Theano flags):
        $ THEANO_FLAGS='floatX=float32,device=gpu,init=True' python run.py v   
        where v \in {1..8}. 

        This will learn the model and predict connections. 
        Will evaluate Normal2 and Normal3 networks after training and print to stdout the AUC scores.
        The stdout are already saved in /stdout directory.
        The script will produce a single Kaggle submission file. (res_normal-1_ver{v}.csv)

        The submission is a simple average of these 8 runs with different seeds + simple normalization.
        Pure output of this script with v==1 gets around 93.679 % AUC on validation network.
        
        
    3.  Averaging and normalization. Output file is the first parameter:
        python mergeNormalize.py OUTPUT LIST_OF_CSV_RESULTS_TO_AVERAGE
        
        This will take around 5 minutes. Run:
        python mergeNormalize.py kaggleSubmission.csv res_ver1.csv res_ver2.csv res_ver3.csv res_ver4.csv res_ver5.csv res_ver6.csv res_ver7.csv res_ver8.csv
            
        kaggleSubmission.csv is a final submission to Kaggle. 
        







    







 