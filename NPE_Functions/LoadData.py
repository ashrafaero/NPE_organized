#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:51:04 2024
@author: akassem
"""

import pandas as pd

def LoadData(A0, Data_Path0, Data_Path_csv, freq):
    AAAs={}
    MMMs={}
    ts={}
    Shapes={}
    for a in A0:
        Path_amps='/a'+str(a)
        filename= Data_Path_csv 
        data = pd.read_csv(Data_Path0+Path_amps+filename, delimiter='\t')
        Shapes[a]=data['Time'].shape[0]
        
        ts[a]=data['Time'][0:Shapes[a]:freq] - data['Time'][0]
        AAAs[a]=data['Pitch'][0:Shapes[a]:freq]
        MMMs[a]= data['Mz_Lab'][0:Shapes[a]:freq]
    return ts, AAAs, MMMs