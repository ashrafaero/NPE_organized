#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:18:20 2024

@author: akassem
"""
import torch
torch.set_default_tensor_type(torch.FloatTensor) # set the default to float32
import pyro
import pyro.distributions as dist

from NPE_Functions import model_test

def model_mcmc(t,y,CMq_interpolator,in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator, N_peaks, up_bound_cmq, low_bound_cmq, Sigma_InGuess):
    # Define our intercept prior
    CMq_s = pyro.sample("Cmq_samples", dist.Uniform(torch.ones(cor_angles.shape[0])*torch.tensor(low_bound_cmq),torch.ones(cor_angles.shape[0])*torch.tensor(up_bound_cmq)))
    # Define our model
    CMq_s=CMq_s.clone().detach()
    mean = model_test.model_test(t,CMq_s,in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator, N_peaks)
    # Define a sigma prior for the random error
    sigma = pyro.sample("sigma", dist.HalfNormal(scale=Sigma_InGuess))
    with pyro.plate("data", y.shape[1]):
        # Assume our expected mean comes from a normal distribution
        outcome_dist = dist.Normal(mean, sigma)
        # Condition the expected mean on the observed target y
        return pyro.sample("obs", outcome_dist, obs=y)
