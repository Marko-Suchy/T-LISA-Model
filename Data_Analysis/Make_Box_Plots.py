#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:16:31 2024

This script is meant to analyze "output_csv" from 2.18.24

Generally, the sceript creates histograms, looking at error grouped by paramater.
We may see some trend in  error distribution by paramater...

@author: markosuchy
"""

#%%
import pandas as pd
from matplotlib import pyplot as plt

#%%Get Output CSV
df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/T-LISA Parameter Sweep 2.20.24/output_csv_2_20_24.csv")

df['avg_err'] = (abs(df['L_avg_err']) + abs(df["I_avg_err"]) + abs(df["S_avg_err"]) + abs(df["A_avg_err"])) / 4

#%%start making histgrams


#plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size


df.boxplot(column = ["avg_err"], by = ["N"])

plt.ylabel("Total Average Error")
plt.xlabel("N")

plt.suptitle('')
plt.title("Error Distribution for Monte-Carlo Simulations on ER Graphs")

plt.tight_layout()

#plt.show()
#plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/N_Error_Distb.png", dpi = 900)



df.boxplot(column = ["avg_err"], by = ["r"])
plt.ylabel("avg_err")

df.boxplot(column = ["avg_err"], by = ["gamma"])
plt.ylabel("avg_err")

df.boxplot(column = ["avg_err"], by = ["omega"])
plt.ylabel("avg_err")

df.boxplot(column = ["avg_err"], by = ["k"])


#plt.title("Distibution of error across paramaters")
plt.ylabel("avg_err") 






