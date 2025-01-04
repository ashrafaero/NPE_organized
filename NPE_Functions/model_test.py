#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:07:25 2024

@author: akassem
"""
import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)
from scipy.integrate import solve_ivp

from NPE_Functions import planar_eom, peak_finder, peak_func

def model_test(tt,CMq_interpolator,in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator, N_peaks):

    # Numpy/torch change -
    # Soon it the code will be completely torch and this chunk will be deleted
    if isinstance(CMq_interpolator[0], float):
        CMq_interpolator = np.array(CMq_interpolator)
    elif isinstance(CMq_interpolator[0], np.float32):
        CMq_interpolator = np.array(CMq_interpolator)
    elif isinstance(CMq_interpolator[0], torch.Tensor):
        if CMq_interpolator[0].requires_grad:
            CMq_interpolator = CMq_interpolator.detach().numpy()
        else:
            CMq_interpolator = np.array(CMq_interpolator)

    CL_alpha_interpolator = cla
    CM_alpha_interpolator = cma

    alpha = {}
    for a in in_angles:
        a0=np.radians(a)
        timepoints = np.array(tt[a])
        # run initial value problem
        rop = solve_ivp(lambda t,y: planar_eom.planar_eom(t, y, CA_interpolator=CA_interpolator, \
                             CL_alpha_interpolator=CL_alpha_interpolator, \
               CMq_interpolator=CMq_interpolator,CM_alpha_interpolator=CM_alpha_interpolator, \
                   props=props, cor_angles=cor_angles,interpolate_loc=interpolate_loc,),\
                [0, timepoints[-1]], [0, a0, 0], method='RK45', \
                t_eval=timepoints, rtol=1e-5,atol=1e-5)
        alpha[a] = rop.y[1]

    coef, _ = peak_finder.peak_finder(tt, in_angles, alpha)
    xtest = np.zeros([in_angles.shape[0],N_peaks])
    j=0
    for a in in_angles:
        t_loc = np.linspace(0, np.array(tt[a])[-1],N_peaks)
        xtest[j,:] = peak_func.peak_func(t_loc, *coef[j,:])
        j += 1
    alpha_peaks = torch.from_numpy(xtest)

    return alpha_peaks
