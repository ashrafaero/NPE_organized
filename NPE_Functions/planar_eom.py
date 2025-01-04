#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:03:06 2024
@author: akassem
"""

import numpy as np

def planar_eom(t, y, CA_interpolator,CL_alpha_interpolator, CMq_interpolator,CM_alpha_interpolator, props,cor_angles,interpolate_loc):
    ''' y is a vector such that alpha dot = eta = y[0], alpha = y[1], x = y[2]
    returns [eta prime=alpha double dot, alpha dot, xdot = velocity] '''

    a0 = props['initial_angle']
    rho  = props['density']
    v0   = props['initial_velocity']
    S    = props['area']
    D    = props['diameter']
    Izz  = props['izz']
    mass = props['mass']
    dofs = props['degrees_of_freedom']

    alpha_curr = abs(y[1])
    # print(y[1])
    # input()
    # account for deceleration if dof>=3
    if dofs>=3:
        CA = CA_interpolator  #CA_interpolator is an interpolator object which returns CA(alpha) when provided an alpha

        tshift = (2*mass)/(rho*CA*v0*S) #equation 20 in RTO-MP-AVT-152B
        velcurr = (2*mass)/(rho*CA*(t+tshift)*S) #equation 18 in RTO-MP-AVT-152B

    else: # 2dof and 1dof do not decelerate
        velcurr=v0

    # turn off CL_alpha if dof=1.
    if dofs>1:
        # CL_alpha_interpolator is an interpolator object which returns CL_alpha(alpha) when provided an alpha
        # can set this to a constant reasonable value for initial studies
        CL_alpha = np.interp(alpha_curr, np.radians(interpolate_loc), CL_alpha_interpolator)

    else:
        CL_alpha=0

    # Interpolate to find CMq
    CMq = np.interp(alpha_curr, np.radians(cor_angles), CMq_interpolator)

    # Interpolate to find CM_alpha
    # CMq_interpolator is an interpolator object which returns CMq(alpha) when provided an alpha
    CM_alpha = np.interp(alpha_curr, np.radians(interpolate_loc), CM_alpha_interpolator)

    # build up the equation of motion (Equation 7 in RTO-MP-AVT-152B)
    const1 = ((rho*velcurr*S)/(2*mass))
    const2 = ((mass*D*D)/(2*Izz))
    f = const1*(-CL_alpha+const2*CMq)


    const3 = ((rho*velcurr*velcurr*S*D)/(2*Izz))
    g = const3*CM_alpha

    return [f*y[0]+g*y[1], y[0], velcurr]
