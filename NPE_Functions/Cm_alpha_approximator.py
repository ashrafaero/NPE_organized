#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:56:09 2024
@author: akassem
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Cm_alpha_approximator(props,in_angles,AAAs,MMMs,interpolate_loc):
    denom = 0.5 * props['density'] * props['initial_velocity']**2 * props['area'] * props['diameter']
    AAA=[]; MMM=[]
    for a in in_angles:
        AAA=np.concatenate((AAA, AAAs[a]))
        MMM=np.concatenate((MMM, MMMs[a]))
    idx = np.argsort(AAA)
    AA = AAA[idx]
    MM = MMM[idx]/denom
    interpolate_loc = np.radians(interpolate_loc)
    data = np.array([AA,MM])
    def func(x, a, b, c):
        return a * np.sin(x)**5 + b * np.sin(x)**3 + c * np.sin(x)
    popt, pcov = curve_fit(func, data[0,:], data[1,:])
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(np.degrees(AA),MM,'.')
    plt.plot(np.degrees(data[0,:]), func(data[0,:], *popt), 'r',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.show()
    print(popt)
    cma = np.cos(interpolate_loc) * (popt[0] * 5 * np.sin(interpolate_loc)**4 + 3 * popt[1] * np.sin(interpolate_loc)**2 + popt[2])
    return cma