#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#@author:  Standard Code
"""
This code has been modified as of 10/16/23 to reflect differential equations based on the new model rules!

2/2/24 - THIS CODE IS THE WORKING CODE FOR THE T-LISA MFT PRIOR TO RESCALING!
"""

# This loads some pacakges that have the ODE integrators
import scipy, scipy.integrate
import pylab
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd


# Parameters
r = .9
gamma = .12
omega = .12
#alpha = 1 + gamma * r  #Alpha is no longer relevant in the redefined model

# Initial condition
L0 = 0
S0 = 0.2
A0 = 0
I0 = 1 - S0 - A0

Y0 = [L0, I0, S0, A0 ]

tMax = 30

# Time vector for solution
T = scipy.linspace(0, tMax, 1001)


# This defines a function that is the right-hand side of the ODEs
# Warning!  Whitespace at the begining of a line is significant!
def rhs(Y, t, omega, gamma, r):
    '''
    SIR model.
    
    This function gives the right-hand sides of the ODEs.
    '''
    
    # Convert vector to meaningful component vectors
    # Note: Indices start with index 0, not 1!
    L = Y[0]
    I = Y[1]
    S = Y[2]  
    A = Y[3]
    # The right-hand sides
    dL = gamma * r * S * I + omega * r * A * I
    dI = -(S+A)*I - gamma * r * S * I - omega * r * A * I
    dS = (S+A)*I - gamma * S
    
    dA = gamma*S
    # Convert meaningful component vectors into a single vector
    dY = [ dL, dI, dS, dA ]

    return dY


# Integrate the ODE
# Also, 'args' passes parameters to right-hand-side function.
solution = scipy.integrate.odeint(rhs,Y0,T,args = (omega, gamma, r))
        
L = solution[:, 0]
I = solution[:, 1]
S = solution[:, 2]
A = solution[:, 3]

print("State at t = " + str(tMax) + ". \nL: " + str(L[len(L) - 1]) + "\nI: " + str(I[len(I)-1]) 
      + "\nS: " + str(S[len(S)-1]) + "\nA: " + str(A[len(A)-1]))
# Make plots

# Load a plotting package
"""

pylab.figure()

pylab.plot(T, L,T, I,T, S, T, A)

pylab.xlabel('Time')
pylab.ylabel('Luddite/Ignorant/Suceptible/Adopted')
pylab.legend([ 'Luddite', 'Ignorant', 'Suceptible', 'Adopted' ])
pylab.title("Redefined LISA model: $\\gamma$ = " + str(gamma) + " $\\omega = $" + str(omega) + " $r$ = " + str(r))

# Actually display the plot
pylab.show() """
#%%plot MFT with MatPLotLib

#Plot the T-LISA
plt.plot(T, L, label = "Luddite", color = "blue")
plt.plot(T, I, label = "Ignorant", color = "green")
plt.plot(T, S, label = "Susceptible", color = "orange")
plt.plot(T, A, label = "Adopted", color = "red")


#Plot stored LISA MFT Data
df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/LISA MFT/LISA Model MFT: $\gamma$ = 0.35 $r$ = 0.9.csv")
plt.plot(df["T"], df['L'], color = "blue", alpha = 0.3)
plt.plot(df["T"], df['I'], color = "green", alpha = 0.3)
plt.plot(df["T"], df["S"], color = "orange", alpha = 0.3)
plt.plot(df["T"], df["A"], color = "red", alpha = 0.3)

#Set plot paramaters
plt.legend()
title = "T-LISA Model MFT: $\\gamma$ = " + str(gamma) + " $\\omega$ = " + str(omega) + " $r$ = " + str(r)
plt.title(title)

plt.xlabel("Time")
plt.ylabel("Population Proportion")



#plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/T-LISA MFT/" + title + ".png", dpi = 900)


#%%Explore Phase Space
#Hmmm thigs are not looking good for when L' is negative! but we try anyway...

#2 nested for loops!

omega_linspace = scipy.linspace(0,1, num = 40) #NOTE OMEGA IS LIMITED HERE TO 0 on the bottom!
gamma_linspace = scipy.linspace(0,1, num = 40)
tMax = 60

#initialize list to store values in - PLOT LUDDITES AT tMax
df = pd.DataFrame(columns=(["omega","gamma", "w*g", "L_tMax"]))


for gamma in gamma_linspace:
    for omega in omega_linspace:
        sol = scipy.integrate.odeint(rhs,Y0,T,args = (omega, gamma, r))
        L_tMax = sol[:, 0][len(sol[:, 0]) - 1]
        df = df.append({"omega": omega, "gamma": gamma,"w*g" : omega*gamma, "L_tMax" : L_tMax}, ignore_index = True)
    


#%% prerpare gamma = omega and omega = 0 lines

#df.drop(df.index[29:], inplace = True)
only_gamma = df.loc[df['omega'] == 0]
gamma_equals_omega = df.loc[df['omega'] == df['gamma']]


plt.scatter(only_gamma['L_tMax'],only_gamma['gamma'],  s = 5, color = "red")

#%% plot some stuff  
plt.clf()

df['g'] = abs(df["omega"])  #Totally red > omega = 1, gamma = 0
df['r'] = abs(df["gamma"]) * 0  #totally green > omega = 0, gamma = 1
df['b'] = df["omega"] * 0
#Red poo

colors = list(zip(df['r'], df['g'], df['b']))


#Scatter possibiloties
plt.scatter(df['gamma'], df['L_tMax'],  s = 5, c = colors)

#PLot LISA Phase Line
LISA_phase_df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/LISA MFT/LISA_Phase_Line.csv")
plt.plot(LISA_phase_df["gamma"], LISA_phase_df["L_tMax"], label = "Orignial LISA Phase Line", color = "orange", lw = 3, linestyle = "--")


#plot the gamma = omega and omega = 0 lines 
plt.plot(only_gamma["gamma"], only_gamma['L_tMax'], label = "$\omega$ = 0", color = "red") #Note only omega is defined below! 
plt.plot(gamma_equals_omega["gamma"], gamma_equals_omega["L_tMax"], label = "$\omega = \gamma$", color = "blue")



#Label
plt.ylabel('$L_{tMax}$')
plt.xlabel('$\gamma$')

plt.title("Phase Plot for Terminal LISA model")
plt.legend()

#plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/T-LISA MFT/phase_plot_3", dpi = 900)













