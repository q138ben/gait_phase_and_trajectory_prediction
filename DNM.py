# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:27:48 2019

@author: binbi
"""

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Activation, Flatten,LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM

from keras.layers import Conv2D

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from numpy import array
from keras.utils import np_utils
from sklearn.utils import class_weight
import numpy as np
import random

from fetchDataset import fetchDataset 
from pykalman import UnscentedKalmanFilter

def CNN(X_train,Y_train,X_test,Y_test,img_rows, img_cols ,class_weights,subject):
    
    batch_size = 64
    nb_classes = 5
    nb_epoch = 100
    
    # input image dimensions
    #img_rows, img_cols = 6,3
    # number of convolutional filters to use,best 4
    nb_filters = 4
    
    # size of pooling area for max pooling,default (2,2)
    pool_size = (2, 2)
    # convolution kernel size, default (3,3)
    kernel_size = (3, 3)
    
    
    
    model = Sequential()
    

#
    model.add(Conv2D(nb_filters, kernel_size,input_shape=(img_rows, img_cols, 1),padding='same'))
    model.add(Activation('relu'))
#    model.add(Conv2D(nb_filters, kernel_size))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=pool_size))
#    model.add(Dropout(0.25))

#    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=pool_size))
#    model.add(Dropout(0.25))




    model.add(Flatten())
#    model.add(Dense(512))
#   model.add(Dense(20))
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    print(model.summary())
    
    
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    
    

    
    
    history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=batch_size, nb_epoch=nb_epoch,verbose=0,callbacks=[es,mc],class_weight=None)
    model = load_model(subject+'_best_model.h5')
    
    return model, history


def MLP_(X_train,Y_train,X_test,Y_test,class_weights,subject):
    
    batch_size = 20
    nb_classes = 5
    nb_epoch = 100
    
    
    
    # define model
    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_MLP_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    

    
    history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=batch_size, nb_epoch=nb_epoch,verbose=0,callbacks=[es,mc],class_weight=None)
    model = load_model(subject+'_best_MLP_model.h5')
    
    
    return model, history





def LSTM_model(X_train,Y_train):
    
    max_features = 1024
    nb_classes = 5
    batch_size=200
    epochs=1

    
    
    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    return model


# split a multivariate sequence into samples
def split_sequences(sequences, *args, **kwargs):
    # check if the input is a list
    if isinstance(sequences,list):
        X, y = list(), list()
        for i in range(len(sequences)):
    		# find the end of this pattern
            end_ix = i + kwargs.get('n_steps')
    		# check if we are beyond the dataset
            if end_ix > len(sequences)-1:
                break
    		# gather input and output parts of the pattern
    
            
            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    
    
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    # check if the input is an array
    elif isinstance(sequences,np.ndarray):
        X, y = list(), list()
        
        if kwargs.get('mode') == 'multi_step':
            for i in range(sequences.shape[0]):
                end_ix = i + kwargs.get('n_steps_in')
                #end_ix = kwargs.get('n_steps_in')*(i+1)
                out_end_ix = end_ix + kwargs.get('n_steps_out')
		# check if we are beyond the sequence
                if out_end_ix > sequences.shape[0]-1:
                    break
		# gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix,-1]
                #seq_x, seq_y = sequences[(kwargs.get('n_steps_in')*i):end_ix, :-1], sequences[end_ix:out_end_ix,-1]
                X.append(seq_x)
                y.append(seq_y)




        if kwargs.get('mode') == 'single_step':   
            for i in range(sequences.shape[0]):
                end_ix = i + kwargs.get('n_steps')
                if end_ix > sequences.shape[0]-1:
                    break
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
                
                X.append(seq_x)
                y.append(seq_y)
            
        return array(X), array(y)

def split_IMU_sequences(sequences, *args, **kwargs):
    # check if the input is a list
    if isinstance(sequences,list):
        X, y = list(), list()
        for i in range(len(sequences)):
    		# find the end of this pattern
            end_ix = i + kwargs.get('n_steps')
    		# check if we are beyond the dataset
            if end_ix > len(sequences)-1:
                break
    		# gather input and output parts of the pattern
    
            
            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    
    
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    # check if the input is an array
    elif isinstance(sequences,np.ndarray):
        X, y = list(), list()
        
        if kwargs.get('mode') == 'multi_step':
            for seq in range(sequences.shape[0]):
                end_ix = seq + kwargs.get('n_steps_in')
                #end_ix = kwargs.get('n_steps_in')*(i+1)
                out_end_ix = end_ix + kwargs.get('n_steps_out')
		# check if we are beyond the sequence
                if out_end_ix > sequences.shape[0]-1:
                    break
		# gather input and output parts of the pattern
                seq_x, seq_y = sequences[seq:end_ix, :-1], sequences[end_ix:out_end_ix,-1]
                #seq_x, seq_y = sequences[(kwargs.get('n_steps_in')*i):end_ix, :-1], sequences[end_ix:out_end_ix,-1]
                X.append(seq_x)
                y.append(seq_y)


        if kwargs.get('mode') == 'single_step':   
            for i in range(sequences.shape[0]):
                end_ix = i + kwargs.get('n_steps')
                if end_ix > sequences.shape[0]-1:
                    break
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
                
                X.append(seq_x)
                y.append(seq_y)
            
        return array(X), array(y)

def LSTM1(X_train,Y_train,n_steps,nb_classes,subject):
    # choose a number of time steps
    nb_epoch=30

    # the dataset knows the number of features, e.g. 2
    n_features = X_train.shape[2]
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#    model.add(Dense(1)) # dense 1 for 'mse' loss
    #model.add(Dense(nb_classes))
    
    
    model.add(Activation('softmax'))
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
 #   model.compile(optimizer='adam', loss='mse') 
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_LSTM_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)

    # fit model
#   model.fit(X_train, Y_train, epochs=20, verbose=1)
    model.fit(X_train, Y_train,validation_split=0.2, epochs=nb_epoch,verbose=0,callbacks=[es,mc])
    
    return model

from keras.layers import RepeatVector
from keras.layers import TimeDistributed

def LSTM1_multi_step(X_train,Y_train,n_steps_in,n_steps_out,nb_classes,subject):
    # choose a number of time steps
    nb_epoch=20

    # the dataset knows the number of features, e.g. 2
    n_features = X_train.shape[2]
    # define model
    model = Sequential()
    #model.add(LSTM(50, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(50, activation= 'relu', input_shape=(n_steps_in, n_features)))
    
    #model.add(LeakyReLU(alpha=0.6))


    #model.add(RepeatVector(n_steps_out))
    #model.add(TimeDistributed(Dense(2)))
    
    model.add(Dense(n_steps_out))
    
    
    #model.add(Activation('softmax'))
    #opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer='adam', loss='mse') 
#    model.compile(loss='categorical_crossentropy',
#                  optimizer= opt,
#                  metrics=['accuracy'])
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_LSTM_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)

    # fit model
#   model.fit(X_train, Y_train, epochs=20, verbose=1)
    model.fit(X_train, Y_train,validation_split=0.2, epochs=nb_epoch,verbose=1,callbacks=[es,mc])
    
    return model






def feature_selection(X,feature_ind):
    
    
    
    #feature_ind = (17,26,35,44)
    
    feature_selec = np.zeros((np.shape(X)[0],len(feature_ind)))
    
    for i, e in enumerate(feature_ind):
        feature_selec[:,i]= X[:,e]
    
    
    return feature_selec

def split_train_test(X,y):
    
    LR_ind = [i for i, e in enumerate(y) if e == 1]
    MS_ind = [i for i, e in enumerate(y) if e == 2]
    TS_ind = [i for i, e in enumerate(y) if e == 3]
    PSw_ind = [i for i, e in enumerate(y) if e ==4]
    Sw_ind = [i for i, e in enumerate(y) if e == 5]
    
    
    LR_ind_diff = np.diff(LR_ind)
    Sw_ind_diff = np.diff(Sw_ind)
    
    
    gait_ind_start = np.array([LR_ind[i+1] for i, e in enumerate(LR_ind_diff) if e > 5] ) # get the index for the start of LR in each gait cycle
    
    gait_ind_end = np.array([Sw_ind[i] for i, e in enumerate(Sw_ind_diff) if e > 5] )  # get the index for the end of Sw in each gait cycle
    
    
    gait_ind_start = np.insert(gait_ind_start,0,0) # insert index 0 
    gait_ind_start = np.delete(gait_ind_start,-1) # delete the  last index becaur it is not a full cycle
    
    random.seed(9) #seed 8,9
    
    test_start = int(0.7*len(gait_ind_start))
    
    test_start_ind = random.randint(0,test_start)
    
    test_end_ind = test_start_ind +len(gait_ind_start)- test_start
    
    
    test_start_ind = gait_ind_start[test_start_ind]
    test_end_ind = gait_ind_end[test_end_ind]
    
    
    X_train = np.concatenate((X[:test_start_ind ,:],X[test_end_ind:,:]),axis=0)
    X_test = X[test_start_ind:test_end_ind,:]
    y_train = np.concatenate((y[:test_start_ind],y[test_end_ind:]),axis = 0)
    y_test = y[test_start_ind:test_end_ind]
    
    return X_train,X_test,y_train,y_test



def fit_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(5, input_dim=63, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(5, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=50, verbose=0)
	return model




def eval_standalone_model(trainX, trainy, testX, testy, n_repeats):
	scores = list()
	for _ in range(n_repeats):
		# define and fit a new model on the train dataset
		model = fit_model(trainX, trainy)
		# evaluate model on test dataset
		_, test_acc = model.evaluate(testX, testy, verbose=0)
		scores.append(test_acc)
	return scores


def eval_standalone_CNN(X_train,Y_train,X_test,Y_test,class_weights,subject, n_repeats):
	scores = list()
	for _ in range(n_repeats):
		# define and fit a new model on the train dataset
		model,history = CNN(X_train,Y_train,X_test,Y_test,class_weights,subject)
		# evaluate model on test dataset
		test_acc = model.evaluate(X_test, Y_test, verbose=0)
		scores.append(test_acc)
	return scores



# repeated evaluation of a model with transfer learning
def eval_transfer_model(trainX, trainy, testX, testy, n_fixed, n_repeats):
	scores = list()
	for _ in range(n_repeats):
		# load model
		model = load_model('model.h5')
		# mark layer weights as fixed or not trainable
		for i in range(n_fixed):
			model.layers[i].trainable = False
		# re-compile model
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		# fit model on train dataset
		model.fit(trainX, trainy, epochs=50, verbose=0)
		# evaluate model on test dataset
		_, test_acc = model.evaluate(testX, testy, verbose=0)
		scores.append(test_acc)
	return scores


# repeated evaluation of a model with transfer learning
def eval_transfer_CNN(trainX, trainy, testX, testy, n_fixed, n_repeats):
    scores = list()
    for _ in range(n_repeats):
		# load model
        model = load_model('D:/data_aquisition/hao/hao_frontal_4.0transferLearning_best_model.h5')
		# mark layer weights as fixed or not trainable
        for i in range(n_fixed):
            model.layers[i].trainable = False
		# re-compile model
        opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss='categorical_crossentropy',optimizer= opt,metrics=['accuracy'])
		# fit model on train dataset
        model.fit(trainX, trainy, epochs=50, verbose=0)
		# evaluate model on test dataset
        _, test_acc = model.evaluate(testX, testy, verbose=0)
        scores.append(test_acc)
        return scores
    
def kalmanFilter(X):
    
     
    ukf = UnscentedKalmanFilter()
    
    X_fil = np.zeros(np.shape(X))
    
    for i in range(np.shape(X)[1]):
        X_fil[:,i] = ukf. smooth(X[:,i])[0].ravel()
        
        
    return X_fil
    
#if __name__ == "__main__":
#    
#    subject = 'qing_frontal_3.7'
#    X,y= fetchDataset(subject)
#    
#    seq_x, seq,y = split_sequences(s, n_steps):
##    X_train,X_test,y_train,y_test = split_train_test(X,y)