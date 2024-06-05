#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:57:29 2024

This code is meant to plot the data produced by multiple runs of the T-LISA simulation! 

@author: markosuchy
"""

#%% Imports
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

#%%Load Data (and drop paramaters)

#This is the data for stochastic simulations
#df = pd.read_json("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/data1_1_23_24.json")
df = pd.read_json("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/data1_2_6_24.json")

params = df["paramaters"][0]
#Note, Params are stored like [N, simulations, k, P, inital_conditions, r, gamma, omega]

df = df.drop(0,0)
df = df.drop("paramaters",1)


#Load MFT data
MFT = pd.read_json("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/MFT_1_24_24.json")
MFT = MFT.drop(['N', 'tMax', 'k', 'P', 'Y0', 'r', 'gamma', 'omega'], axis = 1)

#%%Get averages for the stochastic runs

L_sol_array = np.array([df.loc[i, "L_sol"] for i in [1, 2, 3, 4, 5]])
average_L_sol = np.mean(L_sol_array, axis=0)

I_sol_array = np.array([df.loc[i, "I_sol"] for i in [1, 2, 3, 4, 5]])
average_I_sol = np.mean(I_sol_array, axis=0)

S_sol_array = np.array([df.loc[i, "S_sol"] for i in [1, 2, 3, 4, 5]])
average_S_sol = np.mean(S_sol_array, axis=0)

A_sol_array = np.array([df.loc[i, "A_sol"] for i in [1, 2, 3, 4, 5]])
average_A_sol = np.mean(A_sol_array, axis=0)

df.loc[len(df.index) + 1] = ['average', average_L_sol, average_I_sol, average_S_sol, average_A_sol] 

#%%Start to plot the dang thang!
plt.clf()

fig, ax = plt.subplots()

for i in df.index:
    
    if i == 6: #for plotting the avergaes!
        #plot the average
        plt.plot(df["L_sol"][i], color = "blue", alpha = 0.6, linestyle='dashed')
        plt.plot(df["I_sol"][i], color = "green", alpha = 0.6, linestyle='dashed')
        plt.plot(df["S_sol"][i], color = "orange", alpha = 0.6, linestyle='dashed')
        plt.plot(df["A_sol"][i], color = "red", alpha = 0.6, linestyle='dashed')
        
    else:
        plt.plot(df["L_sol"][i], color = "blue", alpha = 0.2)
        plt.plot(df["I_sol"][i], color = "green", alpha = 0.2)
        plt.plot(df["S_sol"][i], color = "orange", alpha = 0.2)
        plt.plot(df["A_sol"][i], color = "red", alpha = 0.2)
    
    
plt.plot(MFT["T"], MFT["L"]*300, label = "Luddite" ,color = "blue")
plt.plot(MFT["T"], MFT["I"]*300, label = "Ignorant" ,color = "green")
plt.plot(MFT["T"], MFT["S"]*300, label = "Succeptible" ,color = "orange")
plt.plot(MFT["T"], MFT["A"]*300, label = "Adopted" ,color = "red")


#generate some randome stuff to get a good legend
#plt.plot(MFT["T"], MFT["I"]*300, label = "Stochastic Simulation" ,color = "black", alpha = 0.6)
plt.plot(MFT["T"], MFT["I"]*300, label = "Stochastic Average" ,color = "black", linestyle = 'dashed')    


pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(1.75, 0.5))



title = "T-LISA MC vs MFT: " + "N = " + str(params[0]) + ", k = " + str(params[2]) + ", $\gamma{}$ = " + str(round(params[6], 5)) + ", $\omega{}$ = " + str(round(params[7], 5)) + " $r$ = " + str(params[5]) 
plt.title(title)


    
#plt.text(75,50,"L_Error: yatta yatta \n I_Err: yatta")

#plt.show()
#ax.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/Stochastic_vs_MFT_03_22_23_with_full_legend", dpi = 900)

fig_legend = plt.figure(figsize=(3, 2))
ax_legend = fig_legend.add_subplot(111)

ax_legend.axis('off')
# Draw the legend on the new figure
ax_legend.legend(*ax.get_legend_handles_labels(), loc='center')

fig_legend.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/Stochastic_vs_MFT_03_22_23_with_full_legend", dpi = 900, bbox_inches='tight')

#%% Calculate Error




