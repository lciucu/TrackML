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

#string_stage="11111" # all steps
#string_stage="11000" # data
#string_stage="00100" # NN train
#string_stage="00010" # NN analyze
string_stage="00001" # plot

# output stem
inputFolderName="./input"
outputFolderName="./output"

# valueInputOutput=io
list_io="Input,Output".split(",")
# valueTrainTest=tt
list_tt="Train,Test".split(",")
# valueLossAccuracy=la
list_la="Loss,Accuracy".split(",")

dict_io_fileName={}
for io in list_io:
    dict_io_fileName[io]=outputFolderName+"/NN_1_data_"+io+".npy"

dict_io_tt_fileName={}
for io in list_io:
    for tt in list_tt:
        dict_io_tt_fileName[io+tt]=outputFolderName+"/NN_2_data_"+io+tt+".npy"

list_color=["r","b","m","k","g","o"]
    
eventNumber="000021069"

bucketSize=20
nrNodesInputLayer=bucketSize*3 # three inputs (x, y, z) for each hit in the batch
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

if doPlot:
    import matplotlib.pyplot as plt
    
# for DNN training with Keras
# the order is layer and k (from architecture), nrEpoch, batchSize (from learning steps)
list_infoNN=[
    # ["A1",2,10,200],
    ["A1",2,300,200],
    ["A2",2,300,200],
    ["A3",2,300,200],
    ["A1",4,300,200],
    ["A2",4,300,200],
    ["A3",4,300,200],
    ["A1",8,300,200],
    ["A2",8,300,200],
    ["A3",8,300,200],
    ["A1",2,300,1000],
    ["A2",2,300,1000],
    ["A3",2,300,1000],
    ["A1",4,300,1000],
    ["A2",4,300,1000],
    ["A3",4,300,1000],
    ["A1",8,300,1000],
    ["A2",8,300,1000],
    ["A3",8,300,1000],
]
   
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

def get_from_infoNN(infoNN):
    layer=infoNN[0]
    k=infoNN[1]
    nrEpoch=infoNN[2]
    batchSize=infoNN[3]
    if verbose:
        print("Start do NN part for","layer",layer,"k",str(k),"nrEpoch",str(nrEpoch),"batchSize",str(batchSize))
    nameNN="l_"+layer+"_k_"+str(k)+"_e_"+str(nrEpoch)+"_b_"+str(batchSize)
    if debug:
        print("nameNN",nameNN)
    # done if
    return nameNN,layer,k,nrEpoch,batchSize
# done function

def get_dict_la_tt_fileName(nameNN):
    dict_la_tt_fileName={}
    for la in list_la:
        for tt in list_tt:
            dict_la_tt_fileName[la+tt]=outputFolderName+"/NN_3_learn_"+la+tt+"_"+nameNN+".npy"
    dict_la_tt_fileName["nrEpoch"]=outputFolderName+"/NN_3_learn_"+"nrEpoch"+"_"+nameNN+".npy"
    # ready to return
    return dict_la_tt_fileName
# done function

def get_fileNameWeights(nameNN):
    fileNameWeights=outputFolderName+"/NN_3_learn_model_weights_"+nameNN+".hdf5"
    if debug:
        print("fileNameWeights",fileNameWeights)
    # done if
    return fileNameWeights
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
            output=1.0
        else:
            output=0.0
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
        # print 
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
            if io=="Input":
                # reshape to have an extra dimension of size one, needed by Keras
                dict_io_tt_nparray[io+tt]=dict_io_tt_nparray[io+tt].reshape(dict_io_tt_nparray[io+tt].shape[0],dict_io_tt_nparray[io+tt].shape[1],1)
            # print
            print_nparray(io+tt,dict_io_tt_nparray[io+tt])
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
    return dict_io_tt_nparray
# done function

#########################################################################################################
#### Functions for NN training and analyzing
#######################################################################################################

# create an empty NN model, and documentation related to it
# https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/layers/core/
# https://keras.io/activations/
def prepare_NN_model(layer="A3",k=4):
    if debug or verbose:
        print("")
        print("Prepare empty NN model (fixed geometry and weights filled with random numbers).")
    nrNodesHiddenLayer=bucketSize*k # let the user change this k as hyper-parameter
    # create empty model
    model=keras.models.Sequential()
    # define the geometry by defining how many layers and how many nodes per layer
    # add input layer
    # Flatten(): https://stackoverflow.com/questions/44176982/how-flatten-layer-works-in-keras?rq=1
    model.add(keras.layers.Dense(nrNodesInputLayer,activation='linear',input_shape=(nrNodesInputLayer,1)))
    # flatten input layer
    model.add(keras.layers.Flatten())
    # add hidden layers
    if layer=="A1":
        model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
    elif layer=="A2":
        model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
        model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
    elif layer=="A3":
        model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
        model.add(keras.layers.Dense(nrNodesHiddenLayer,activation='relu'))
        model.add(keras.layers.Dense(nrNodesOutputLayer,activation='relu'))
    else:
        print("layer",layer,"not known. Choose A1,A2,A3. WILL ABORT!!!")
        assert(False)
    # done if
    # add output layer
    model.add(keras.layers.Dense(nrNodesOutputLayer,activation='sigmoid'))
    # finished defining the geometry, and now define how the NN learns (is trained)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # done all, ready to return
    return model
# done function

def train_NN_model(dict_io_tt_nparray,model,nameNN,nrEpoch,batchSize):
    if verbose:
        print("Start train NN for",nameNN)
    # fit the model
    for io in list_io:
        for tt in list_tt:
            print_nparray(io+" "+tt,dict_io_tt_nparray[io+tt])
    # done forloop
    h=model.fit(
        dict_io_tt_nparray["Input"+"Train"],dict_io_tt_nparray["Output"+"Train"],
        batch_size=batchSize,epochs=nrEpoch,verbose=1,
        validation_data=(dict_io_tt_nparray["Input"+"Test"],dict_io_tt_nparray["Output"+"Test"]),
        shuffle=False
    )
    if debug:
        print("h.history")
        print(h.history)
        print("h.history.keys()")
        print(h.history.keys())
        print("print(h.history['val_loss'])")
        print(h.history['val_loss'],type(h.history['val_loss']))
    # losses and accuracy
    dict_la_tt_nparray={
         "Loss"+"Train":np.array(h.history['loss']),
         "Loss"+"Test":np.array(h.history['val_loss']),
         "Accuracy"+"Train":np.array(h.history['accuracy']),
         "Accuracy"+"Test":np.array(h.history['val_accuracy']),
         "nrEpoch":np.array(range(nrEpoch)),
     }
    #
    dict_la_tt_fileName=get_dict_la_tt_fileName(nameNN)
    # write to .npy files
    for key in dict_la_tt_nparray.keys():
        np.save(dict_la_tt_fileName[key],dict_la_tt_nparray[key])
    # done for loop
    # write learned weights of the model to .hdf5
    fileNameWeights=get_fileNameWeights(nameNN)
    model.save_weights(fileNameWeights)
    # finished all tasks, nothing to return
# done function

    
def analyze_NN_model(nameNN,model,io,tt):
    if verbose:
        print("Start train NN for",nameNN)
    fileNameWeights=get_fileNameWeights(nameNN)
    model.load_weights(fileNameWeights)
    if verbose:
        print("  End train NN for",nameNN)
# done function


########################################################################################################
#### Functions about plotting
#######################################################################################################

def overlayGraphsValues(list_tupleArray,outputFileName="overlay",extensions="pdf,png",info_x=["Procent of data reduced",[0.0,1.0],"linear"],info_y=["Figure of merit of performance",[0.0,100000.0],"log"],info_legend=["best"],title="Loss and Accuracy",debug=False):
    if debug:
        print("Start overlayGraphsValues")
        print("outputFileName",outputFileName)
        print("extensions",extensions)
        print("info_x",info_x)
        print("info_y",info_y)
        print("info_legend",info_legend)
        print("title",title)
    # x axis
    x_label=info_x[0]
    x_lim=info_x[1]
    x_lim_min=x_lim[0]
    x_lim_max=x_lim[1]
    if x_lim_min==-1 and x_lim_max==-1:
        x_set_lim=False
    else:
        x_set_lim=True
    x_scale=info_x[2]
    if debug:
        print("x_label",x_label,type(x_label))
        print("x_lim_min",x_lim_min,type(x_lim_min))
        print("x_lim_max",x_lim_max,type(x_lim_max))
        print("x_set_lim",x_set_lim,type(x_set_lim))
        print("x_scale",x_scale,type(x_scale))
    # y axis
    y_label=info_y[0]
    y_lim=info_y[1]
    y_lim_min=y_lim[0]
    y_lim_max=y_lim[1]
    if y_lim_min==-1 and y_lim_max==-1:
        y_set_lim=False
    else:
        y_set_lim=True
    y_scale=info_y[2]
    if debug:
        print("y_label",y_label,type(y_label))
        print("y_lim_min",y_lim_min,type(y_lim_min))
        print("y_lim_max",y_lim_max,type(y_lim_max))
        print("y_set_lim",y_set_lim,type(y_set_lim))
        print ("y_scale",y_scale,type(y_scale))
    # create empty figure
    plt.figure(1)
    # set x-axis
    plt.xlabel(x_label)
    if x_set_lim==True:
        plt.xlim(x_lim_min,x_lim_max)
    plt.xscale(x_scale)
    # set y-axis
    plt.ylabel(y_label)
    if y_set_lim==True:
        plt.ylim(y_lim_min,y_lim_max)
    plt.yscale(y_scale)
    # set title
    plt.title(title)
    # fill content of plot
    for i,tupleArray in enumerate(list_tupleArray):
        if debug:
            print("i",i,"len",len(tupleArray))
        x=tupleArray[0]
        y=tupleArray[1]
        c=tupleArray[2]
        l=tupleArray[3]
        plt.plot(x,y,c,label=l)
    # done loop over each element to plot
    # set legend
    plt.legend(loc=info_legend[0])
    # for each extension create a plot
    for extension in extensions.split(","):
        fileNameFull=outputFileName+"."+extension
        print("Saving plot at",fileNameFull)
        plt.savefig(fileNameFull)
    # close the figure
    plt.close()
# done function()

def test_plot():
    list_tupleArray=[]
    nparray_x=np.load(dict_la_tt_fileName["nrEpoch"])
    nparray_y=np.load(dict_la_tt_fileName["Accuracy"+"Train"])
    print_nparray("x-nrEpoch",nparray_x)
    print_nparray("y-AccuracyTrain",nparray_y)
    color="r"
    legendName="Accuracy Train"
    list_tupleArray.append((nparray_x,nparray_y,color,legendName))
    outputFileName=outputFolderName+"/NN_plot1D_optionTrainTest_"+"Accuracy"
    extensions="png,pdf"
    plotRange=[-1,-1]
    overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions=extensions,
                        info_x=["Number of epochs",[-1,-1],"linear"],
                        info_y=["Value of the "+"Accuracy"+" function",plotRange,"linear"],
                        info_legend=["best"],title="NN_"+"Accuracy",debug=False)
    
# done function()

def plot_Loss_Accuracy(nameNN):
    dict_la_tt_fileName=get_dict_la_tt_fileName(nameNN)
    for la in list_la:
        list_tupleArray=[]
        nparray_x=np.load(dict_la_tt_fileName["nrEpoch"])
        print_nparray("x-nrEpoch",nparray_x)
        for i,tt in enumerate(list_tt):
            nparray_y=np.load(dict_la_tt_fileName[la+tt])
            print_nparray("y-"+la+tt,nparray_y)
            color=list_color[i]
            legendName=la+" "+tt
            list_tupleArray.append((nparray_x,nparray_y,color,legendName))
        # done for loop
        outputFileName=outputFolderName+"/NN_plot1D_optionTrainTest_"+la+"_"+nameNN
        extensions="png,pdf"
        plotRange=[-1,-1]
        overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions=extensions,
                            info_x=["Number of epochs",[-1,-1],"linear"],
                            info_y=["Value of the "+la+" function",plotRange,"linear"],
                            info_legend=["best"],title="NN_"+la,debug=False) 
        # done for loop    
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
    if doNNTrain or doNNAnalyze:
        dict_io_tt_nparray=read_from_file_NN_data_dict_io_tt_nparray()
        # loop over different NN that we compare (arhitecture and learning)
        for infoNN in list_infoNN:
            nameNN,layer,k,nrEpoch,batchSize=get_from_infoNN(infoNN)
            # create empty train model architecture (layer and k), with bad initial random weights
            model=prepare_NN_model(layer,k)
            if doNNTrain:
                train_NN_model(dict_io_tt_nparray,model,nameNN=nameNN,nrEpoch=nrEpoch,batchSize=batchSize)
            if doNNAnalyze:
                pass
            # done if
        # done for loop over infoNN
    # done if
    if doPlot:
        # loop over different NN that we compare (arhitecture and learning)
        for infoNN in list_infoNN:
            nameNN,layer,k,nrEpoch,batchSize=get_from_infoNN(infoNN)
            plot_Loss_Accuracy(nameNN)
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
