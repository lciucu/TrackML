#!/usr/bin/env python

# import from basic Python to be able to read automatically the name of a file
import sys
import os

# import to use numpy arrays
import numpy as np
# to obtain reproducible results, meaning each new NN training to obtain the same result, set the seed of the random number now
# np.random.seed(2019)
# np.random.seed(20190825)
np.random.seed(98383822)

#########################################################################################################
#### configuration options
#########################################################################################################

debug=False
verbose=True
doTest=False

string_stage="01000" # all steps

# output stem
inputFolderName="./input"
outputFolderName="./output"
fileNameNNInput=outputFolderName+"/NN_input.npy"
fileNameNNOutput=outputFolderName+"/NN_output.npy"

#eventNumber="000000032"
eventNumber="000021069"

bucketSize=20
k=2
nrNodesInputLayer=bucketSize*3 # three inputs (x, y, z) for each hit in the batch
nrNodesHiddenLayer=bucketSize*k # let the user change this k as hyper-parameter
nrNodesOutputLayer=bucketSize*1 # one output for each hit in the batch
nrHitsTest=bucketSize*4 # 2 in test 2 in train


list_stage=list(string_stage)
doNNInputOutput=bool(int(list_stage[0]))
doNNInputOutputTrainTest=bool(int(list_stage[1]))
doNNTrain=bool(int(list_stage[2]))
doNNAnalyze=bool(int(list_stage[3]))
doPlot=bool(int(list_stage[4]))

if debug:
    print("string_stage",string_stage)
    print("list_stage",list_stage)
    print("doNNInputOutput",doNNInputOutput)
    print("doNNInputOutputTrainTest",doNNInputOutputTrainTest)
    print("doNNTrain",doNNTrain)
    print("doNNAnalyze",doNNAnalyze)
    print("doPlot",doPlot)

if doNNInputOutput:
    import pandas as pd
    from collections import Counter

if doNNTrain or doNNAnalyze:
    import keras

#########################################################################################################
#### Functions general
#######################################################################################################

# a general function to print the values and other properties of a numpy array
# use to see the values of the numpy arrays in our code for debugging and understanding the code flow
def print_nparray(name,nparray):
    if verbose or debug:
        print("")
        print("nparray",name)
        print(nparray)
        print("type",type(nparray),"shape",nparray.shape,"min_value=%.3f"%np.min(nparray),"min_position=%.0f"%np.argmin(nparray),"max_value=%.3f"%np.max(nparray),"max_position=%.0f"%np.argmax(nparray))
# done function

def print_df(name,df):
    if verbose or debug:
        print(name,"shape",df.shape)
        # print(df.head())
        # print(df.tail())
        print(df)
# done function

#########################################################################################################
#### Functions for creaing numpy arrays of input and output for NN training
#########################################################################################################

# df means panda data frame
def get_df_from_file(name,inputFileName):
    if debug:
        print("name",name,"inputFileName",inputFileName)
    df=pd.read_csv(inputFileName)
    print_df(name,df)
    return df
# done function

def get_df_hits_for_one_event(eventNumber):
    df_hits_recon=get_df_from_file("df_hits_recon",inputFolderName+"/event"+eventNumber+"-hits.csv")
    df_hits_truth=get_df_from_file("df_hits_truth",inputFolderName+"/event"+eventNumber+"-truth.csv")
    df_particles=get_df_from_file("df_particles",inputFolderName+"/event"+eventNumber+"-particles.csv")
    # combine df_hits_recon and df_hits_truth into a common df_hits
    df_hits=pd.concat([df_hits_recon,df_hits_truth],axis=1,sort=False)    
    print_df("df_hits",df_hits)
    return df_hits
# done function

def get_list_inputValues(df_bucket,debug):
    list_inputValues=[]
    for i in range (df_bucket.shape[0]):
        hit=df_bucket.iloc[i]
        x=hit["x"]
        y=hit["y"]
        z=hit["z"]
        if debug:
            print("i",i,"x",x,"y",y,"z",z)
        list_inputValues.append(x)
        list_inputValues.append(y)
        list_inputValues.append(z)
    return list_inputValues
# done function

def get_particle_id_most_common(df_bucket,debug):
    counter=Counter(df_bucket.particle_id.values)
    list_tuple_most_common=counter.most_common()
    tuple_most_common=list_tuple_most_common[0]
    particle_id_most_common=tuple_most_common[0]
    if debug:
        print("particle_id_most_common",particle_id_most_common)
    return particle_id_most_common
# done function

def get_list_outputValues(df_bucket,debug):
    list_outputValues=[]
    particle_id_most_common=get_particle_id_most_common(df_bucket,debug)
    for i in range (df_bucket.shape[0]):
        hit=df_bucket.iloc[i]
        particle_id=hit["particle_id"]
        if particle_id==particle_id_most_common:
            output=1
        else:
            output=0
        if debug and False:
            print("i", i,"particle_id",particle_id,"particle_id_most_common",particle_id_most_common,"output",output)
        list_outputValues.append(output)
    # done for loop
    return list_outputValues
# done function

def write_NN_input_output_to_files(df_hits,doWriteOnlyAnEvenNumberOfBuckets):
    nrHits=df_hits.shape[0]
    if debug:
        print("nrHits",nrHits)
    if doTest:
        nrHits=nrHitsTest 
        print("doTest=true, so hacking nrHits to be a small number("+str(nrHitsTest)+"), so that we run just a quick test")
    if doWriteOnlyAnEvenNumberOfBuckets:
        nrBuckets=(nrHits-nrHits%bucketSize)/bucketSize
        if debug:
            print("nrBuckets",nrBuckets)
        nrBuckets=nrBuckets-nrBuckets%2
        if debug:
            print("nrBuckets",nrBuckets)
    # done if
    counterBuckets=0
    list_list_inputValues=[]
    list_list_outputValues=[]
    for i in range (nrHits):
        isMultipleofBucketSize=(i%bucketSize==0)
        if isMultipleofBucketSize==False:
            continue
        isCompleteBucket=i+bucketSize<=nrHits
        if isCompleteBucket==False:
            continue
        if doWriteOnlyAnEvenNumberOfBuckets:
            isNotLastOddBucket=counterBuckets<nrBuckets
            if isNotLastOddBucket==False:
                continue
        # done if
        counterBuckets+=1
        if debug or (verbose and counterBuckets%100==0):
            print(i,isMultipleofBucketSize,counterBuckets)
        df_bucket=df_hits[i:i+bucketSize]
        list_inputValues=get_list_inputValues(df_bucket,debug)
        if debug:
            print("list_inputValues",list_inputValues)
        list_outputValues=get_list_outputValues(df_bucket,debug)
        if debug:
            print("list_outputValues", list_outputValues)
        list_list_inputValues.append(list_inputValues)
        list_list_outputValues.append(list_outputValues)
    # done for loop
    if debug:
        print("counterBuckets",counterBuckets)
        print("length_input",len(list_list_inputValues))
        print("length_output",len(list_list_outputValues))
    # convert lists to numpy arrays
    nparray_input=np.array(list_list_inputValues)
    if debug or verbose:
        print("nparray_input", nparray_input.shape, nparray_input.dtype)
        print(nparray_input)
    nparray_output=np.array(list_list_outputValues)
    if debug or verbose:
        print("nparray_output", nparray_output.shape, nparray_output.dtype)
        print(nparray_output)
    # save numpy arrays to files
    np.save(fileNameNNInput,nparray_input)
    np.save(fileNameNNOutput,nparray_output)
# done function

#########################################################################################################
#### Functions for reading back the numpy arrays from files and split them into train and test
#########################################################################################################

# only one function for train and test, to be called twice, once for train, once for test
def get_nparray_train_test(name,fileNameNNInputOrOutput):
    if debug:
        print("get_nparray_train_test for name",name,"fileNameNNInputOrOutput",fileNameNNInputOrOutput)
    nparray=np.load(fileNameNNInputOrOutput)
   # print(nparray)
   # print(nparray.shape)
    print_nparray(name,nparray)
    return 1,2
# done function

#########################################################################################################
#### Functions for NN training and analyzing
#######################################################################################################

# create an empty NN model, and documentation related to it
# https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/layers/core/
# https://keras.io/activations/
def prepare_NN_model(bucketSize,k):
    # create empty model
    model=keras.models.Sequential()
    # define the geometry by defining how many layers and how many nodes per layer
    # add input layer
    model.add(keras.layers.Dense(nrNodesInputLayer,activation='linear',input_shape=(nrNodesInputLayer,1)))
    # flatten input layer
    model.add(keras.layers.Flatten())
    # add hidden layers
    model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
    model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
    model.add(keras.layers.Dense(nrNodesOutputLayer,activation='relu'))
    # add output layer
    model.add(keras.layers.Dense(nrNodesOutputLayer,activation='sigmoid'))
    # finished defining the geometry, and now define how the NN learns (is trained)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # done all, ready to return
    return model
# done function

########################################################################################################
#### Function doAll() putting all together
#######################################################################################################

# the function that runs everything
def doItAll():
    if doNNInputOutput:
        df_hits=get_df_hits_for_one_event(eventNumber)
        write_NN_input_output_to_files(df_hits,doWriteOnlyAnEvenNumberOfBuckets=False)
    if doNNInputOutputTrainTest:
        nparray_input_train,nparray_input_test=get_nparray_train_test("input",fileNameNNInput)
        #nparray_output_train,nparray_output_test=get_nparray_train_test("output",fileNameNNOutput)
    if doNNTrain:
        model=prepare_NN_model(bucketSize,k)
    # done if
# done function

#########################################################################################################
#### Run all
#######################################################################################################

doItAll()

#########################################################################################################
#### Done
#######################################################################################################

print("")
print("")
print("Finished well in "+sys.argv[0]) # prints out automatically the name of the file that we ran

exit()
