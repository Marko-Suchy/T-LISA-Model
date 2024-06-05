#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 21:31:45 2024

@author: markosuchy

This file is meant to join structural network variables with run outputs!
Then, we use OLS regression to look to see if network-level paramaters can be used a a predictor 
of A_tMax_prop
"""

import pandas as pd

import json
from networkx.readwrite import json_graph

import os

import networkx as nx

import statsmodels.api as sm

from matplotlib import pyplot as plt

import random
#%%

data_folder = "/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/3_22_24/Data"
graph_folder = "/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/3_22_24/Final_Graphs"


df = pd.DataFrame(columns=['N', 'simulations', 'k', 'inital_conditions', 'r', 'gamma', 'omega', 'Simulation_batch_num', 'L_tMax', 'A_tMax', 'atr_assort_coef', 'transitivity', 'avg_clustering', 'avg_shortest_path_length', 'top_nodes_A_prop', "dominating_set_A_prop", "random_ten_pct_A_prop", "random_set_A_prop"])

iterer = 0
for filename in os.listdir(data_folder):
    
    
    
    
    #Get file paths
    data_path = os.path.join(data_folder, filename)
    graph_path = os.path.join(graph_folder, filename)
    
    #Get stuff from data
    data_df = pd.read_json(data_path)
    L_tMax = int(data_df.tail(1)["L_sol"])
    A_tMax = int(data_df.tail(1)["A_sol"])
    
    #Start the new row by putting params dict in it
    new_row = dict(data_df.head(1)['params'])[0] #
    new_row['L_tMax'] = L_tMax
    new_row['A_tMax'] = A_tMax
    
    
    
    #get graph
    with open(graph_path, 'r') as f:
        data = json.load(f)
        
    G = json_graph.node_link_graph(data)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    '''
    #Start getting graph level characteristics
    atr_assort_coef = nx.attribute_assortativity_coefficient(G, 'state')
    new_row['atr_assort_coef'] = atr_assort_coef
    
    new_row['transitivity'] = nx.transitivity(G)
    
    new_row['avg_clustering'] = nx.average_clustering(G)
    
    new_row['avg_shortest_path_length'] = nx.average_shortest_path_length(G)
    
    #new_row['eigenvector_centrality'] = nx.eigenvector_centrality(G)
    '''
    #Next idea - look at the ratio of the top 10% of nodes by voterank, than see if that's
    #a predictor of final outcome
    top_node_set = nx.voterank(G, number_of_nodes = int(0.1*len(G)))
    top_node_states = [G.nodes[node]['state'] for node in top_node_set]
    top_node_states_dummy = [1 if state == "A" else 0 for state in top_node_states]
    new_row['top_nodes_A_prop'] = sum(top_node_states_dummy) / len(top_node_states_dummy)
    
    #dominating set proportion
    dominating_set = list(nx.dominating_set(G))
    dominating_set_states = [G.nodes[node]['state'] for node in dominating_set]
    dominating_set_states_dummy = [1 if state == "A" else 0 for state in dominating_set_states]
    new_row['dominating_set_A_prop'] = sum(dominating_set_states_dummy) / len(dominating_set_states_dummy)
    
    #Random set, len 10%
    random_ten_pct = random.sample(list(G.nodes()), len(top_node_set))
    random_ten_pct_states = [G.nodes[node]['state'] for node in random_ten_pct]
    random_ten_pct_states_dummy = [1 if state == "A" else 0 for state in random_ten_pct_states]
    new_row['random_ten_pct_A_prop'] = sum(random_ten_pct_states_dummy) / len(random_ten_pct_states_dummy)
    
    #random set, len = dominating set len
    random_dom_set = random.sample(list(G.nodes()), len(dominating_set))
    random_dom_set_states = [G.nodes[node]['state'] for node in random_dom_set]
    random_dom_set_states_dummy = [1 if state == "A" else 0 for state in random_dom_set_states]
    new_row['random_set_A_prop'] = sum(random_dom_set_states_dummy) / len(random_dom_set_states_dummy)
    
    
    
    
    
    #add row to dataframe
    df.loc[len(df)] = new_row
    
    

    
    #track progess
    iterer = iterer + 1
    print(str(iterer) + "/30,000")

    

#%%Get rid of nan Exog
#df.to_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/3_22_24/joint_graph_and_state_data.csv")
#df.to_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/3_22_24/joint_graph_and_state_data_set_stuff.csv")
df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/3_22_24/joint_graph_and_state_data.csv")


#df = df.dropna()

df["P"] = df['k'] / (df["N"] - 1)

df["L_tMax_prop"] = df["L_tMax"] / df["N"]
df["A_tMax_prop"] = df["A_tMax"] / df["N"]

#%%Look at data
df.boxplot("A_tMax_prop")
plt.show()
df.boxplot("atr_assort_coef")



#%%OLS
X = df[['gamma', 'omega', 'r']]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit() #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#%%
X = df[['gamma', 'omega', 'r', "N", "k"]]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit() #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#%%



#%%Null Model! R^2 is pretty good!
X = df[['gamma', 'omega', 'r', "N", "k"]]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#%%
X = df[['gamma', 'omega', 'r', "N", "k", "top_nodes_A_prop"]]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit() #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#%%including assortivity and other stuff
X = df[['gamma', 'omega', 'r', "N", "k", "atr_assort_coef", "transitivity", "avg_clustering", "avg_shortest_path_length"]]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary().as_latex())


#%%including assortivity and other stuff
X = df[['gamma', 'omega', 'r', "top_nodes_A_prop"]]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') #default for cov_type is nonrobust ... maybe robust is better?
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())


#%% SET ANALYSIS
X = df[['gamma', 'omega', 'r']]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') 
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())


#%% votetank set
#null (random) set
X = df[['gamma', 'omega', 'r', 'top_nodes_A_prop']]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') 
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#voterank set
X = df[['gamma', 'omega', 'r', 'random_ten_pct_A_prop']]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') 
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#%%
#null (random) set
X = df[['gamma', 'omega', 'r', 'dominating_set_A_prop']]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') 
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())

#voterank set
X = df[['gamma', 'omega', 'r', 'random_set_A_prop']]
X = sm.add_constant(X)

y = df['A_tMax_prop']

model = sm.OLS(y, X)
results = model.fit(cov_type = 'HC0') 
#Use cov_type='HC0' for robust covarriance. 

# Print the summary results
print(results.summary())







        