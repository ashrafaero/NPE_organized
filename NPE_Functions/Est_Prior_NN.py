#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:23:39 2024

@author: akassem
"""
import numpy as np
from os.path import exists
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from NPE_Functions import estimate_priori

def Est_Prior_NN(props,t,MMs,in_angles,CMq_InGuess_NN,cor_angles,ni_count,interpolate_loc,cma,cla, CA_interpolator, N_peaks, Rerun_NN, N_neurons, epochs, batch_size, validation_split, Spacing_cmq_NN):
    
    if exists('In_Out_NN.npy') and Rerun_NN==0:
        with open('In_Out_NN.npy', 'rb') as f2:
            xtest = np.load(f2)
            X = np.load(f2)
            Y = np.load(f2)
            n_count = np.load(f2)
    else:
        xtest, ps, ue, n_count = estimate_priori.estimate_priori(props,t,MMs,in_angles,CMq_InGuess_NN,cor_angles,ni_count,interpolate_loc,cma,cla, CA_interpolator, Spacing_cmq_NN, N_peaks)
        # Copy the input and output data for DNN
        xtest, X, Y, n_count = np.copy(xtest), np.copy(ps), np.copy(ue), n_count
        # Save
        with open('In_Out_NN.npy', 'wb') as f2:
            np.save(f2, np.array(xtest))
            np.save(f2, np.array(X))
            np.save(f2, np.array(Y))
            np.save(f2, n_count)

    ind = np.linspace(0, n_count-1, n_count, dtype=int)
    ind_train, ind_test, _, _ = train_test_split(ind, ind, test_size=0.6, random_state=42)

    xtrain, xtest_case, ytrain, ytest = X[ind_train], X[ind_test], Y[ind_train], Y[ind_test]

    nf, nl = xtrain.shape[1], ytrain.shape[1]
    # Define the layers and neurons
    inp = Input(shape=(nf,))
    xx = Dense(N_neurons, activation="relu")(inp)
    xx = Dense(N_neurons, activation="relu")(xx)
    xx = Dense(N_neurons, activation="relu")(xx)
    xx = Dense(N_neurons, activation="relu")(xx)
    xx = Dense(N_neurons, activation="relu")(xx)
    output = Dense(nl, activation="linear")(xx)

    model_NN = Model(inp, output)
    model_NN.summary()
    model_NN.compile(loss='mse', optimizer='adam')
    model_NN.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,validation_split=validation_split)

    # model_NN.save('model.h5')

    # model_NN.compile(loss=asymmetric_loss(alpha), optimizer='adam')
    
    # i=0
    # for a in in_angles:
    #     t_loc = np.linspace(0, np.array(t[a])[-1],N_peaks)
    #     plt.plot(t_loc,xtest[i,:],'-o')
    #     i += 1
    # plt.show()
    
    M_test = xtest.flatten().reshape([1,N_peaks*in_angles.shape[0]])
    Cmq_pred = model_NN.predict(M_test)
    
    return Cmq_pred
