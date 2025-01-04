#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:09:59 2024

@author: akassem
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from NPE_Functions import peak_func, peak_func_lin

def peak_finder(t,in_angles,data,type='parabola'):
    if type=='parabola':
        coef,pcov = np.zeros([in_angles.shape[0],3]), np.zeros([in_angles.shape[0],3,3])
    elif type=='linear':
        coef,pcov = np.zeros([in_angles.shape[0],2]), np.zeros([in_angles.shape[0],2,2])
    i=0
    for a in in_angles:
        peaks, _ = find_peaks(abs(np.array(data[a])), height=0)
        if peaks[0]!=0 and peaks[0]!=1:
            peaks = np.insert(peaks, 0, 0)
        
        est_peaks = abs(np.array(data[a]))[peaks]
        est_peaktimes = np.array(t[a])[peaks]
        
        if type=='parabola':
            coef[i,:], pcov[i,:,:] = curve_fit(peak_func.peak_func, est_peaktimes, est_peaks)
        elif type=='linear':
            coef[i,:], pcov[i,:,:] = curve_fit(peak_func_lin.peak_func_lin, est_peaktimes, est_peaks)
        i += 1    
    return coef,pcov
