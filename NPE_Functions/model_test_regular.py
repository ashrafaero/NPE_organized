#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:14:52 2024

@author: akassem
"""

import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)
from scipy.integrate import solve_ivp

from NPE_Functions import planar_eom

def model_test_regular(tt,CMq_interpolator,in_angles,cor_angles,props,interpolate_loc,cma,cla, CA_interpolator):

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
    return alpha
