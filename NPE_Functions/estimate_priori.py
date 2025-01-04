#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:21:22 2024

@author: akassem
"""

import numpy as np
import matplotlib.pyplot as plt

from NPE_Functions import peak_finder, peak_func, model_test_moment

def estimate_priori(props,t,MMs,in_angles,CMq_InGuess_NN,cor_angles,ni_count,interpolate_loc,cma,cla, CA_interpolator, Spacing_cmq_NN, N_peaks):

    coef, _ = peak_finder.peak_finder(t, in_angles, MMs)
    xtest = np.zeros([in_angles.shape[0],N_peaks])
    
    j=0
    for a in in_angles:
        t_loc = np.linspace(0, np.array(t[a])[-1],N_peaks)
        xtest[j,:] = peak_func.peak_func(t_loc, *coef[j,:])
        plt.plot(t_loc,xtest[j,:],'-o')
        plt.plot(np.array(t[a]),np.array(MMs[a]),'-')
        j += 1
    plt.show()
    
    n_count = 1
    mod_vec = np.zeros(CMq_InGuess_NN.shape[0])
    mod_vec[0] = ni_count[-1]
    for i in range(ni_count.shape[0]):
        n_count *= ni_count[i]
        if i!=ni_count.shape[0]-1:
            mod_vec[i+1] = mod_vec[i]*ni_count[-1-i-1]
    mod_vec = np.flip(mod_vec)

    ue = np.zeros((n_count, cor_angles.shape[0]))
    ps = np.zeros((n_count, in_angles.shape[0]*N_peaks))
    counter = 0
    iii = np.zeros(mod_vec.shape[0])
    print('Trajectory Generation is Starting...')
    print('Total Trajectory: ',n_count)
    
    for i in range(n_count):
        if i%100==0: print('Current Trajectory: ',i)
        CMq_InGuess_NN_j = CMq_InGuess_NN-iii*Spacing_cmq_NN
        iii[-1] += 1
        for ii in np.linspace(mod_vec.shape[0]-1,1,mod_vec.shape[0]-1,dtype=int):
            if iii[ii]==ni_count[ii]:
                iii[ii] = 0
                iii[ii-1] += 1
        ue[counter,:] = CMq_InGuess_NN_j
        
        alpha_dot, alpha, alpha_T, x, V = model_test_moment.model_test_moment(t,CMq_InGuess_NN_j,in_angles,props,cor_angles,interpolate_loc,cma,cla, CA_interpolator)
        
        M_total={}
        for a in in_angles:
            denom2 = 0.5 * props['density'] * V[a]**2 * props['area'] * props['diameter']
            cmq_term = alpha_dot[a] * props['diameter'] / (2 * V[a])
            M_dynamic = denom2 * np.interp(alpha_T[a], np.radians(cor_angles), ue[counter,:]) * cmq_term 	
            M_static = denom2 * np.interp(abs(alpha[a]), np.radians(interpolate_loc), cma)  * alpha[a]  
            M_total[a] = M_static + M_dynamic
        
        coef, _ = peak_finder(t, in_angles, M_total)
        M_total_peaks = np.zeros([in_angles.shape[0],N_peaks])
        j=0
        for a in in_angles:
            t_loc = np.linspace(0, np.array(t[a])[-1],N_peaks)
            M_total_peaks[j,:] = peak_func(t_loc, *coef[j,:])
            j += 1
            
        ps[counter,:] = M_total_peaks.flatten()
        counter += 1
        
    return xtest, ps, ue, n_count
