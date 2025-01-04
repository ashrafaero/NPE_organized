#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:16:10 2024

@author: akassem
"""

import numpy as np
import matplotlib.pyplot as plt

def subplot(time, ymean, y_up, y_dwn, ytrue,labelMean, N_seed, labeltrue, fileName,in_angles, Zoomed_Traj, Z_pts):
    
    # Full
    for a in in_angles:
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.plot(np.array(time[a]), np.degrees(np.array(ytrue[a])),'-', label=labeltrue, linewidth = 4)
        ax.plot(np.array(time[a]), np.degrees(np.array(ymean[a])),'--o', label=labelMean, linewidth = 2,markevery=1)
        ax.set_ylabel(r'$\alpha\:[deg]$', labelpad=5,fontsize=26)
        ax.set_xlabel(r'$Time\:[s]$',fontsize=26)
        ax.set_title('seed = {}'.format(N_seed),fontsize=26)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=22)
        fig.tight_layout()
        plt.savefig(fileName+str(a)+'deg_seed{}.png'.format(N_seed), dpi = 500, bbox_inches = 'tight')
        plt.show()
        fig.clear(True)
    # Zoomed
    if Zoomed_Traj==1:
        for a in in_angles:
            fig, ax = plt.subplots(1, 1, figsize=(8,6))
            ax.plot(np.array(time[a])[:Z_pts], np.degrees(np.array(ytrue[a])[:Z_pts]),'-', label=labeltrue, linewidth = 4) # First: [:Z_pts]   Last: [-Z_pts:-1]
            ax.plot(np.array(time[a])[:Z_pts], np.degrees(np.array(ymean[a])[:Z_pts]),'--o', label=labelMean, linewidth = 2,markevery=1)
            ax.set_ylabel(r'$\alpha\:[deg]$', labelpad=5,fontsize=22)
            ax.set_xlabel(r'$Time\:[s]$',fontsize=18)
            ax.set_title('seed = {}'.format(N_seed),fontsize=26)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=18)
            fig.tight_layout()
            plt.savefig(fileName+str(a)+'deg_Zoomed_seed{}.png'.format(N_seed), dpi = 500, bbox_inches = 'tight')
            plt.show()
            fig.clear(True)
            