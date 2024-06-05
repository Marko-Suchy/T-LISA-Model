#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:41:58 2024

@author: markosuchy

This plot is meant to regress error on various paramaters to see if error is 
DATA SOURCE is 

"""

#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
#%%
df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/T-LISA Parameter Sweep 2.20.24/output_csv_2_20_24.csv")

#make avg error column
df['avg_err'] = (abs(df['L_avg_err']) + abs(df["I_avg_err"]) + abs(df["S_avg_err"]) + abs(df["A_avg_err"])) / 4

#%%set respnse variable - avg error
y = df['avg_err']  # Dependent variable


#%% First model with only N and K included

print("MODEL WITH ONLY N AND K")

X = df[['N', 'k']] 
X = sm.add_constant(X)


model = sm.OLS(y, X)
results = model.fit() #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#print(results.summary().as_latex())

#%% Run model with more paramaters included.

print("MODEL WITH MORE PARAMATERS")

X = df[['N', 'r', 'gamma', 'omega', 'k']]  
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

# Print the summary results
print(results.summary())
#print(results.summary().as_latex())

#Export model to LaTex
#print("LATEX CODE STARTS HERE ----- ")
#print(results.summary().as_latex())

#%% Run thw mdoel with interacgion terms included

print("MODEL WITH INTERACTION TERM")

df['N_k_interaction'] = df['N'] * df['k']

X = df[['N', 'r', 'gamma', 'omega', 'k', 'N_k_interaction']]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

# Print the summary results
print(results.summary())
#print(results.summary().as_latex())

#%%model with ln(N)

df['ln_N'] = np.log(df['N'])

X = df[['ln_N', 'r', 'gamma', 'omega', 'k']]  
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

# Print the summary results
print(results.summary())





