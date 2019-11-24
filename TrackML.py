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

string_stage="11100" # all steps

# output stem
inputFolderName="./input"
outputFolderName="./output"

# valueInputOutput=io
list_io="Input,Output".split(",")
# valueTrainTest=tt
list_tt="Train,Test".split(",")

dict_io_fileName={}
for io in list_io:
    dict_io_fileName[io]=outputFolderName+"/NN_data_1_"+io+".npy"

dict_io_tt_fileName={}
for io in list_io:
    for tt in list_tt:
        dict_io_tt_fileName[io+tt]=outputFolderName+"/NN_data_2_"+io+tt+".npy"
    
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

def write_to_file_NN_data_dict_io_nparray(df_hits):
    nrHits=df_hits.shape[0]
    if debug:
        print("nrHits",nrHits)
    if doTest:
        nrHits=nrHitsTest 
        print("doTest=true, so hacking nrHits to be a small number("+str(nrHitsTest)+"), so that we run just a quick test")
    # done if
    counterBuckets=0
    dict_io_list_list_value={}
    for i in range (nrHits):
        isMultipleofBucketSize=(i%bucketSize==0)
        if isMultipleofBucketSize==False:
            continue
        isCompleteBucket=i+bucketSize<=nrHits
        if isCompleteBucket==False:
            continue
        if debug or (verbose and counterBuckets%100==0):
            print(i,isMultipleofBucketSize,counterBuckets)
        counterBuckets+=1
        df_bucket=df_hits[i:i+bucketSize]
        #
        dict_io_list_value={}
        dict_io_list_value["Input"]=get_list_inputValues(df_bucket,debug)
        dict_io_list_value["Output"]=get_list_outputValues(df_bucket,debug)
        for io in list_io:
            if debug:
                print("list_value "+io,dict_io_list_value[io]) 
            if io not in dict_io_list_list_value:
                dict_io_list_list_value[io]=[]
            else:
                dict_io_list_list_value[io].append(dict_io_list_value[io])
            # done if
        # done for loop over io
    # done for loop over hits
    if debug:
        print("counterBuckets",counterBuckets)
    # done if
    dict_io_nparray={}
    for io in list_io:
        print("length_"+io,len(dict_io_list_list_value[io]))
        # convert list_value to nparray
        dict_io_nparray[io]=np.array(dict_io_list_list_value[io])
        print_nparray(io,dict_io_nparray[io])
        # write nparray to file
        np.save(dict_io_fileName[io],dict_io_nparray[io])
    # done for loop over io
# done function

#########################################################################################################
#### Functions for reading back the numpy arrays from files and split them into train and test
#########################################################################################################

def write_to_file_NN_data_dict_io_tt_nparray():
    dict_io_tt_nparray={}
    for io in list_io:
        nparray=np.load(dict_io_fileName[io])
        print_nparray(io,nparray)
        nrRow=nparray.shape[0]
        dict_tt_list_index={}
        dict_tt_list_index["Train"]=[i for i in range(nrRow) if i%2==0] # even indices
        dict_tt_list_index["Test"] =[i for i in range(nrRow) if i%2==1] # odd  indices
        for tt in list_tt:
            dict_io_tt_nparray[io+tt]=nparray[dict_tt_list_index[tt],:]
            # write nparray to file
            np.save(dict_io_tt_fileName[io+tt],dict_io_tt_nparray[io+tt])
        # done for loop over tt
    # done for loop over io
# done function

def read_from_file_NN_data_dict_io_tt_nparray():
    dict_io_tt_nparray={}
    for io in list_io:
        for tt in list_tt:
            dict_io_tt_nparray[io+tt]=np.load(dict_io_tt_fileName[io+tt])
            print_nparray(io+tt,dict_io_tt_nparray[io+tt])
        # done for loop over tt
    # done for loop over io
    return dict_io_tt_nparray[io+tt]       
# done function

#########################################################################################################
#### Functions for NN training and analyzing
#######################################################################################################

# create an empty NN model, and documentation related to it
# https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/layers/core/
# https://keras.io/activations/
def prepare_NN_model(bucketSize,k):
    if debug or verbose:
        print("")
        print("Prepare empty NN model (fixed geometry and weights filled with random numbers).")
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
        write_to_file_NN_data_dict_io_nparray(df_hits)
    if doNNInputOutputTrainTest:
        write_to_file_NN_data_dict_io_tt_nparray()
    if doNNTrain:
        dict_io_tt_nparray=read_from_file_NN_data_dict_io_tt_nparray()
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