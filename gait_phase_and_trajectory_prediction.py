# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:25:24 2020

@author: binbi
"""

from keras.models import Model
from keras.layers import Dense,Input
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import np_utils
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score
import numpy as np
import random
from DNM import split_IMU_sequences
from fetchDataset import fetchDataset


def get_data(subject,*args,**kwargs):
    X,y= fetchDataset(subject)
    if X.shape[1] != 63:
        X= X[:,8:] # exclude other channels e.g. EMG if dataset has more than 63 channels
    y = np.subtract(y,1) # the original y has values from 1-5, so subtract 1 to make it from 0-4
    
    
    if args:
        data = []
        for arg in args:
            print ('arg = ', arg)
            X_imu = X[:,arg]
            data.append(X_imu)
        X = np.array(data).T
    
    
    
    if kwargs.get('mode') == 'gait_phase':
        y = y.reshape(len(y),1)
        
        data = np.concatenate((X,y),axis=1)
    if kwargs.get('mode') == 'IMU':
        data = []
        for arg in args:
            print ('arg = ', arg)
            X_imu = X[:,arg]
            data.append(X_imu)
        data = np.array(data).T
        y = X[:,kwargs.get('to_forecast_chan')].reshape(len(y),1)
        data = np.concatenate((data,y),axis=1)
        
    return data


def train_test_sequence(data, n_steps_in, n_steps_out,split = 0.7):
    
    # split the data into train and test set according to splitsize
    
    num = data.shape[0]
    
    splitSize = int(split*num)
    random.seed(10) #seed 8,9
    splitIndStart = random.randint(0,splitSize)
    
    splitIndEnd = splitIndStart+ num-splitSize
    
    dataTr = np.concatenate((data[:splitIndStart,:],data[splitIndEnd:,:]),axis=0)
    dataTe = data[splitIndStart:splitIndEnd,:]

    
    # generate data sequence for LSTM
    
    X_train,y_train =split_IMU_sequences(dataTr,mode = 'multi_step', n_steps_in = n_steps_in, 
                                     n_steps_out = n_steps_out)
    X_test,y_test =split_IMU_sequences(dataTe,mode = 'multi_step', n_steps_in = n_steps_in, 
                                   n_steps_out = n_steps_out)  

    return X_train,y_train, X_test,y_test
   


def multi_variable_output_classification(X_train,y_train,neurons,n_steps_in, 
                              n_steps_out,n_epoch,batch_size):
    
    n_features = X_train.shape[2]
    
    y_train= np_utils.to_categorical(y_train, 5)
    
    y_list = []
    weight_list = []
    for i in range(n_steps_out):
        if n_steps_out > 1:
            y = y_train[:,i,:]
        else:
            y = y_train
        y_list.append(y)

        weight = 0.7**i
        #weight = 1
        weight_list.append(weight)
    
    

        
    
    input_ = Input(shape=(n_steps_in, n_features))
    
    x = LSTM(neurons,return_state= False,
             kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0))(input_)
    
    out_list = []
    for j in range(n_steps_out):
        out = Dense(5,name='out_%d'%j,activation="softmax",kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0))(x)
        out_list.append(out)
        

    
    model = Model(input_,out_list)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'],loss_weights=weight_list)
#    model.fit(X_train, [y_1,y_2],validation_split=0.2, epochs=n_epoch, 
#              batch_size = batch_size, verbose = 1,sample_weight={'out_1': w1, 'out_2': w2})
    
    es = EarlyStopping(monitor='val_out_1_loss', mode='min', verbose=0,patience = 5)
    mc = ModelCheckpoint('_best_LSTM_multiple_output_model.h5', monitor='val_out_1_acc', mode='max', 
                         verbose=1, save_best_only=True)
   
    
    model.fit(X_train, y_list,validation_split=0.2, epochs=n_epoch, 
              batch_size = batch_size, verbose = 0,callbacks=[es,mc])

    return model




def model_multi_output_evaluate(model, X_test,y_test,n_steps_out):
    
    
    y_pred = model.predict(X_test)
    
    
    
    if n_steps_out > 1:

        y_pred_list = []
        for y in y_pred:
            y_arg = np.argmax(y,axis=1).T
            y_pred_list.append(y_arg)
            
        y_pred = np.array(y_pred_list).T
        acc = accuracy_score(y_test.flatten(),y_pred.flatten())
        
        
        return y_pred,acc
    else:
        y_pred = np.argmax(y_pred,axis=1)
        acc = accuracy_score(y_test,y_pred)
        return y_pred,acc


   
    
    
if __name__ == "__main__":
    

    """
    left thigh: 0-8
    left shank: 9-17
    left ankle: 18-26
    right thigh: 27-35
    right shank: 36-44
    right ankle: 45-53
    pelvis: 54-62
    
    
    """

    
    subject = 'qing_frontal_3.7'
#    data= get_data(subject,49,51,mode= 'IMU', to_forecast_chan = 50)
#    data= get_data(subject,49,40,31,22,13,4,mode='gait_phase')
    data= get_data(subject,mode='gait_phase')
#    
    n_steps_in, n_steps_out = 10,2
    X_train,y_train, X_test,y_test = train_test_sequence(data, n_steps_in, n_steps_out)
    
    
    batch_size = 50
    n_epoch = 50
    neurons = 5*n_steps_out
    


    model = multi_variable_output_classification(X_train,y_train,neurons,n_steps_in, n_steps_out,n_epoch,batch_size)


    y_pred,acc = model_multi_output_evaluate(model, X_test, y_test,n_steps_out)

   
#    acc = accuracy_score(y_test.flatten(),y_pred.flatten())