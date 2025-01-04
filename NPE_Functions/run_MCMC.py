#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:26:14 2024

@author: akassem
"""
import numpy as np

import torch
torch.set_default_tensor_type(torch.FloatTensor) # set the default to float32
import pyro
from pyro.infer import MCMC, NUTS #, Predictive
import time

from NPE_Functions import peak_finder, peak_func, model_mcmc


def run_MCMC(t,AAs,Cmq_pred,in_angles,cor_angles,interpolate_loc,props,num_samples,num_warmup,cma,cla, CA_interpolator, N_peaks, up_bound_cmq, low_bound_cmq, Sigma_InGuess, Sigma_prior, max_tree_depth):
    coef, _ = peak_finder.peak_finder(t, in_angles, AAs)
    xtest = np.zeros([in_angles.shape[0], N_peaks])
    j=0
    for a in in_angles:
        t_loc = np.linspace(0, np.array(t[a])[-1], N_peaks)
        xtest[j,:] = peak_func.peak_func(t_loc, *coef[j,:])
        j += 1
    xobs = torch.from_numpy(xtest)
    X_train_torch = xobs.clone().detach()
    
    wu = num_warmup
    sn = num_samples

    # Clear the parameter storage
    pyro.clear_param_store()

    # my_kernel = NUTS(model_mcmc.model_mcmc, max_tree_depth=3,step_size=2e-2,adapt_step_size=False,init_strategy=pyro.infer.autoguide.initialization.init_to_sample() ) # a shallower tree helps the algorithm run faster
    my_kernel = NUTS(model_mcmc.model_mcmc, max_tree_depth=max_tree_depth) # a shallower tree helps the algorithm run faster
    # num_chain = 8

    initialparams = dict(Cmq_samples=torch.tensor(Cmq_pred),sigma=torch.tensor(Sigma_prior))
    my_mcmc1 = MCMC(my_kernel, num_samples=sn, warmup_steps=wu, initial_params=initialparams ) 

    # Let's time our execution as well
    start_time = time.time()

    # Run the sampler
    my_mcmc1.run(t, X_train_torch,Cmq_pred,in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator, N_peaks, up_bound_cmq, low_bound_cmq, Sigma_InGuess)
    end_time = time.time()
    print(f'Inference ran for {round((end_time -  start_time)/60.0, 2)} minutes')
    print(my_mcmc1.summary())

    return my_mcmc1.get_samples()
