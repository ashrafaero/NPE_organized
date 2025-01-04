#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:30:08 2024

@author: akassem
"""
# Packages:
import numpy as np
from scipy.interpolate import CubicSpline
from os.path import exists
import torch
torch.set_default_tensor_type(torch.FloatTensor) # set the default to float32
import pyro
import matplotlib.pyplot as plt
import random

# NPE Functions:
from NPE_Functions import LoadData, Cm_alpha_approximator, Est_Prior_NN, \
    run_MCMC, model_test_regular, subplot

#%% INPUTS
A0 = [1,5,10,30]  # Initial AOA   #[1,2,5,10,20,30] 
Mach = [1.44]
Data_Path0 = "Mach 1.44"
Data_Path_csv = '/Body_1_Dynamics_Data.csv' 
freq1 = 1
freq2 = 1000
interpolate_loc = np.array([0,10,20,30]) # Alpha Interpolation location for the static estimation (cma, cla, ...)
CA_points = [1.6,1.5,1.2,1.1] # Only needed for dof>=3
N_peaks = 10 # Number of peaks
# Model Properties
props={}
props['initial_angle'] = np.array(A0)
props['density'] = 1.2130 #kg/m^2
props['mach_number'] = 1.44
props['initial_velocity'] = 343.11*props['mach_number'] #m/s
props['area'] = 0.00385 #m^2
props['diameter'] = 0.06942 #m
props['izz'] = 1.99*10**-4 #kg-m^2
props['mass'] = 0.633; #kg
props['degrees_of_freedom'] = 1
# NN Inputs
CMq_InGuess_NN = np.array([3.0, 3.0, 3.0, 3.0]) # Initial Guess of Cmq
cor_angles_cons = np.array([0, 5, 10, 30]) # np.array([1,2,5,10,20,30]) # same size of A0
ni_count = np.array([10, 10, 10, 10]) # same size of in_angles
Rerun_IG = False  # If it's needed to recompute the initial guess of Cmq
Rerun_NN = False  # If it's needed to rerun the NN
Spacing_cmq_NN = 0.4 # Spacing resolution for the NN sampling
N_neurons = 256 # Number of neurons
epochs=1000; batch_size=64; validation_split=0.1
# MCMC Inputs
cor_angles = np.array([0, 5, 7.5, 10, 12.5, 20, 30])  #cor_angles = np.array([0,5,10,20,30])
num_samples= 10000 
num_warmup = 500
Resample_MCMC = False # If it's needed to resample for Cmq
low_bound_cmq = -1; up_bound_cmq = 3 # Uniform distribution 
Sigma_InGuess = 2 # Half Normal Dist
Sigma_prior= 0.1
max_tree_depth = 4 # a shallower tree helps the algorithm run faster
# Define seed for reproducebility
seed = 500 # 10  42  123  789 
N_nsamples = 5 # Number of number of Samples

File_npy = 'cmq_mcmc_{}_samples'.format(num_samples)+'_wn{}'.format(num_warmup)+'_upbound{}'.format(up_bound_cmq) +'_seed{}.npy'.format(seed)
#File_png = ''
File_png = '_sn{}'.format(num_samples)+'_wn{}'.format(num_warmup)+'_seed{}.png'.format(seed)
# plotting inputs
STD_Mult = 2 # STD multiplier
Zoomed_Traj=True # Do you need to zoom in for the last few points in alpha traj
Z_pts = 150
d_alpha = 0.1
###################################### MAIN #####################################
#%% Cma, Cla, and CA Estimation
in_angles = np.array(A0)
ts, AAAs, MMMs = LoadData.LoadData(A0, Data_Path0, Data_Path_csv, freq1)   
cma = Cm_alpha_approximator.Cm_alpha_approximator(props,in_angles,AAAs,MMMs,interpolate_loc)
# cla = CL_alpha_approximator(props,in_angles,AAs,LLs,interpolate_loc)
cla = cma
CA_interpolator = CubicSpline(np.radians(interpolate_loc), CA_points, bc_type='natural')
#%% Loading Data 
ts, AAAs, MMMs = LoadData.LoadData(A0, Data_Path0, Data_Path_csv, freq2)

#%% Initial Guess of Cmq from a NN:
if exists('Cmq_pred.npy') and Rerun_IG==0:
    with open('Cmq_pred.npy', 'rb') as f:
        Cmq_pred = np.load(f)
else:    
    Cmq_pred = Est_Prior_NN.Est_Prior_NN(props,ts,MMMs,in_angles,CMq_InGuess_NN,cor_angles_cons,ni_count,interpolate_loc,cma,cla, CA_interpolator, N_peaks, Rerun_NN, N_neurons, epochs, batch_size, validation_split, Spacing_cmq_NN)
    with open('Cmq_pred.npy', 'wb') as f:
        np.save(f, Cmq_pred)

Cmq_pred = np.array([1.62,-0.51, -0.24, -0.38]) 
Cmq_pred = Cmq_pred.flatten()
print("Cmq_pred NN ", Cmq_pred)                      
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(cor_angles_cons,Cmq_pred,'--b', label='NN', linewidth=3)
Cmq_pred = np.interp(cor_angles, cor_angles_cons, Cmq_pred) # check if :  [1.62, −0.51, −0.36, −0.24, −0.31, −0.35, −0.38] 
plt.plot(cor_angles,Cmq_pred,'-ok', label='NN Interpolated')
plt.legend()
plt.grid('on')
plt.show()
#%% Run MCMC 
random.seed(seed)
np.random.seed(seed)
#tf.random.set_seed(seed)
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

print('No. of seeds', seed)
#  
if exists(File_npy) and Resample_MCMC==0:
    with open(File_npy, 'rb') as fs:
        Cmq_obs = np.load(fs)
else:
    samples = run_MCMC.run_MCMC(ts,AAAs,Cmq_pred,in_angles,cor_angles,interpolate_loc,props,num_samples,num_warmup,cma,cla, CA_interpolator, N_peaks, 
                       up_bound_cmq, low_bound_cmq, Sigma_InGuess, Sigma_prior, max_tree_depth)
    
    Cmq_obs = samples['Cmq_samples'].numpy()
    with open(File_npy, 'wb') as fs:
        np.save(fs, Cmq_obs)

# Setting different samples
N_samp = np.linspace(0.1*num_samples,num_samples-0.1*num_samples,N_nsamples)
Cmq_0 = np.array(Cmq_obs)[:int(N_samp[0]), :]
Cmq_1 = np.array(Cmq_obs)[:int(N_samp[1]), :]
Cmq_2 = np.array(Cmq_obs)[:int(N_samp[2]), :]
Cmq_3 = np.array(Cmq_obs)[:int(N_samp[3]), :]
Cmq_4 = np.array(Cmq_obs)[:int(N_samp[4]), :]

Cmq_full = np.array(Cmq_obs)
Cmq_obs_mean =  np.array([Cmq_0.mean(axis=0), Cmq_1.mean(axis=0), Cmq_2.mean(axis=0), Cmq_3.mean(axis=0), Cmq_4.mean(axis=0)])
Cmq_obs_std =   np.array([Cmq_0.std(axis=0),  Cmq_1.std(axis=0),  Cmq_2.std(axis=0),  Cmq_3.std(axis=0),  Cmq_4.std(axis=0)])      
#%% Getting the Cmq samples. mean, and SD
up = Cmq_obs_mean + STD_Mult*Cmq_obs_std
down = Cmq_obs_mean - STD_Mult*Cmq_obs_std

print('Cmq Initial Guess:')
print(CMq_InGuess_NN)
print('Cmq Mean:')
print(Cmq_obs_mean)
print('Cmq STD:')
print(Cmq_obs_std)

# Plotting the results:
samp_color = ['r','c','m','y','g']
A_points = np.arange(0,cor_angles[-1]+d_alpha,d_alpha)

cmvalmean = np.zeros((Cmq_obs_mean.shape[0], A_points.shape[0]))
cmvalup = np.zeros((Cmq_obs_mean.shape[0], A_points.shape[0]))
cmvaldown = np.zeros((Cmq_obs_mean.shape[0], A_points.shape[0]))
#
for i in range(Cmq_obs_mean.shape[0]):
    CMq_interpolator = CubicSpline(cor_angles, Cmq_obs_mean[i], bc_type='clamped')
    CMq_interpolator_up = CubicSpline(cor_angles, up[i], bc_type='clamped')
    CMq_interpolator_down = CubicSpline(cor_angles, down[i], bc_type='clamped')
    cmvalmean[i] = CMq_interpolator(A_points)
    cmvalup[i]= CMq_interpolator_up(A_points)
    cmvaldown[i]= CMq_interpolator_down(A_points)
    
plt.figure(figsize=(8, 6), dpi = 500)
for i in range(Cmq_obs_mean.shape[0]):
    plt.plot(A_points, cmvalmean[i,:], samp_color[i]+'--', label='{} s'.format(N_samp[i]), linewidth = 2)
    plt.plot(A_points, cmvalup[i,:], samp_color[i]+'--', linewidth = 0.5)
    plt.plot(A_points, cmvaldown[i,:], samp_color[i]+ '--', linewidth = 0.5)
    plt.fill_between(A_points,cmvalup[i,:],cmvaldown[i,:],facecolor=samp_color[i], alpha=0.15, label='{}-STD'.format(STD_Mult))
plt.plot(cor_angles,Cmq_pred,'-ok', label='NN prediction')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlim([0, 21])
plt.ylim([-1.5, 2.5])
plt.grid('on')
plt.ylabel(r'$C_{mq}+C_{m_{\dot{\alpha}}}$',fontsize=26) #, labelpad=5
plt.xlabel(r'$\alpha\:[deg]$',fontsize=26)
plt.legend(fontsize=10, loc = 'upper right') #bbox_to_anchor=(1, 1) center
plt.tight_layout()
plt.savefig('Cmq_different_samples'+File_png, dpi = 500, bbox_inches = 'tight')
plt.show()

#%% Considering the full No. of samples:
Cmq_mean=Cmq_full.mean(axis=0)
Cmq_std=Cmq_full.std(axis=0)
up_f = Cmq_mean+STD_Mult*Cmq_std
down_f = Cmq_mean-STD_Mult*Cmq_std

CMq_interpolator = CubicSpline(cor_angles, Cmq_mean, bc_type='clamped')
CMq_interpolator_up = CubicSpline(cor_angles, up_f, bc_type='clamped')
CMq_interpolator_down = CubicSpline(cor_angles, down_f, bc_type='clamped')
cmvalmean = CMq_interpolator(A_points)
cmvalup= CMq_interpolator_up(A_points)
cmvaldown= CMq_interpolator_down(A_points)

plt.figure(figsize=(8, 6), dpi = 500)
plt.plot(A_points, cmvalmean, 'b'+'--', label='{} s'.format(num_samples), linewidth = 2)
plt.plot(A_points, cmvalup, 'b'+'--', linewidth = 0.5)
plt.plot(A_points, cmvaldown, 'b'+ '--', linewidth = 0.5)
# plt.plot(cor_angles,Cmq_pred,'-ok', label='NN Interp')
plt.fill_between(A_points,cmvalup,cmvaldown,facecolor='b', alpha=0.15, label='{}-STD'.format(2));
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlim([0, 30])
plt.ylim([-1.5, 1.8])
plt.grid('on')
plt.ylabel(r'$C_{mq}+C_{m_{\dot{\alpha}}}$',fontsize=26) #, labelpad=5
plt.xlabel(r'$\alpha\:[deg]$',fontsize=26)
plt.title('{} Samples'.format(num_samples),fontsize=22)
plt.legend(fontsize=12, loc = 'upper right') #bbox_to_anchor=(1, 1) center
plt.tight_layout()
plt.savefig('Cmq_full_samples'+File_png, dpi = 500, bbox_inches = 'tight')
plt.show()    
#%% Recontructing alpha trajectory given Cmq_mean and Cmq_std
for i in range(Cmq_obs_mean.shape[0]):
    xobsmean = model_test_regular.model_test_regular(ts,Cmq_obs_mean[i],in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator)
    x_up = model_test_regular.model_test_regular(ts,up[i],in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator)
    x_dwn = model_test_regular.model_test_regular(ts,down[i],in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator)
    
    # Trajectories plot
    labelMean = "Reconstructed: "+'{} s'.format(int(N_samp[i]))
    N_seed = seed
    labeltrue = "True"
    #fileID1 = 'M-1-44-Cmq-'
    fileID2 = 'Trajectory_'+'{}s_'.format(int(N_samp[i]))
    subplot.subplot(ts, xobsmean, x_up, x_dwn, AAAs, labelMean, N_seed, labeltrue, fileID2,in_angles, Zoomed_Traj, Z_pts)
    