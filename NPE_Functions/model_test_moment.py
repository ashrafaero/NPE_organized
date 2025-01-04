#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:04:33 2024
@author: akassem
"""

import numpy as np
from scipy.integrate import solve_ivp

from NPE_Functions import planar_eom

def model_test_moment(tt,CMq_interpolator,in_angles,props,cor_angles,interpolate_loc,cma,cla, CA_interpolator):
    # define aero coefficients; best to do this in radians
    CL_alpha_interpolator = cla
    CM_alpha_interpolator = cma
    
    alpha_dot = {}
    alpha = {}
    alpha_T = {}
    x = {}
    V = {}
    # define trajectory conditions
    for a in in_angles:
        a0=np.radians(a)
        timepoints = np.array(tt[a])
        # run initial value problem
        rop = solve_ivp(lambda t,y: planar_eom.planar_eom(t, y, CA_interpolator=CA_interpolator, CL_alpha_interpolator=CL_alpha_interpolator, \
               CMq_interpolator=CMq_interpolator,CM_alpha_interpolator=CM_alpha_interpolator, props=props, cor_angles=cor_angles,interpolate_loc=interpolate_loc,),\
                [0, timepoints[-1]], [0, a0, 0], method='RK45', \
                t_eval=timepoints, rtol=1e-5,atol=1e-5)
        # Extract values from the ivp
        alpha_dot[a] = rop.y[0]
        alpha[a] = rop.y[1]
        alpha_T[a] = abs(rop.y[1])
        x[a] = rop.y[2]
        
        #finite difference to get velocity
        vv = np.array([np.diff(x[a])/np.diff(np.array(tt[a]))])[0]; 
        V[a] = np.concatenate((np.array([vv[0]]), np.array(vv)))
    return alpha_dot,alpha,alpha_T,x,V
