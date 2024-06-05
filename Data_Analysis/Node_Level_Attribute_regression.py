#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:21:43 2024

@author: markosuchy

This file is meant to run T-LISA simulations for preparing regression data
 to regress state outcome on node level attributes

"""

import networkx as nx
import matplotlib.pyplot as plt
import random 
import pandas as pd

import scipy, scipy.integrate
import numpy as np

import json
from networkx.readwrite import json_graph

import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


#%%Define ER Graph Class
#I'll add a method here to export a dataframe with each node in the Graph, along with it's final state. 
class ER_Graph():
    def __init__(self, N, simulations, k, inital_conditions, r, gamma, omega):
        self.N = N
        self.gamma = gamma
        self.omega = omega
        self.r = r
        self.simulations = simulations
        self.k = k
        self.inital_conditions = inital_conditions
        
        #Iitialize a bunch of lists that will be plotted 
        self.L_solution = []
        self.I_solution = []
        self.S_solution = []
        self.A_solution = []
        
        #initialize a flags list
        self.breakdown_conditon_flags = []
        
        P = k/(N-1)
        self.P = P
        self.G = nx.erdos_renyi_graph(N, P)
        
        state_list = ['L', 'I', 'S', 'A']
        initial_state = random.choices(state_list, weights=inital_conditions, k = N)
        nx.set_node_attributes(self.G, dict(zip(self.G.nodes(), initial_state)), 'state')
    
        
    def simulation_step(self):
        for n in range(self.N):
            neighbors = list(self.G[n].keys())
            
            #Get the noode n's state
            node_n_state = nx.get_node_attributes(self.G, "state").get(n)
            
            #Get lists of suceptible neighbors
            suceptible_neighbors = []
            adopted_neighbors = []
            for neighbor in neighbors:
                if nx.get_node_attributes(self.G, "state").get(neighbor) == 'S':
                    suceptible_neighbors.append(neighbor)
                elif nx.get_node_attributes(self.G, "state").get(neighbor) == 'A':
                    adopted_neighbors.append(neighbor)
            
                        
            #Flag model breakdown
            #if (self.gamma * len(suceptible_neighbors) + self.omega * len(adopted_neighbors))*(1/len(neighbors)) + (len(suceptible_neighbors) + len(adopted_neighbors))*(1/self.N) >= 1:
            #    self.breakdown_conditon_flags.append(1)
            
            
            
            #Define propensities
            #IN THIS VERSION OF CODE, ADOPTED NODES ARE FACTORED IN!
            S_to_A = self.gamma
            #I_to_L = r*gamma*(len(suceptible_neighbors)+len(adopted_neighbors))/len(neighbors)# - Pre Nov. 2023 update
            #This if-else block was added 2/18/24
            if len(neighbors) > 0:
                I_to_L = self.r*(self.gamma*len(suceptible_neighbors) + self.omega*len(adopted_neighbors))/len(neighbors)
            else:
                I_to_L = 0
            I_to_S = (len(suceptible_neighbors)+len(adopted_neighbors))/self.N

            if node_n_state == "S":
                
                #Define possible new states
                possible_states = ["S", "A"] 
                
                #choose a state based on probability
                state_assignment = random.choices(possible_states, weights=(1 - S_to_A, S_to_A), k = 1)
                nx.set_node_attributes(self.G, {n:state_assignment[0]}, 'state')

                

                #Assign that state to the node at hand
                
            if node_n_state == "I":
                possible_states = ["I", "L", "S"] 
                
                state_assignment = random.choices(possible_states, weights=(1 - (I_to_L + I_to_S), I_to_L, I_to_S), k = 1)
                nx.set_node_attributes(self.G, {n:state_assignment[0]}, 'state')
            

        
    def run_one_simulation(self, animate = False):
        #re-initialize the graph, so states are all re-randomized.
        self.__init__(self.N, self.simulations, self.k, self.inital_conditions, self.r, self.gamma, self.omega)
        
       #make sure solutions lists are empty
        self.L_solution = []
        self.I_solution = []
        self.S_solution = []
        self.A_solution = []
        
       #Run the simulation:
        for simulation in range(self.simulations):
            
            #Actuallly do the simulation, and print progress
            self.simulation_step()
            
            if (simulation + 1) %10 == 0:
                print("ran simulation step:", simulation + 1, "/", self.simulations)
            
            #Store the amount of number of each node in a lists, which will then be plotted
            S_nodes = [n for n,v in self.G.nodes(data=True) if v['state'] == 'S']
            self.S_solution.append(len(S_nodes))
            
            L_nodes = [n for n,v in self.G.nodes(data=True) if v['state'] == 'L']  
            self.L_solution.append(len(L_nodes))
            
            I_nodes = [n for n,v in self.G.nodes(data=True) if v['state'] == 'I']  
            self.I_solution.append(len(I_nodes))
            
            A_nodes = [n for n,v in self.G.nodes(data=True) if v['state'] == 'A']
            self.A_solution.append(len(A_nodes))
            
            if self.check_stable_state():
                print(f"STABLE STATE REACHED at step {simulation + 1}")
                break   
            
            if animate:
                self.draw_graph()
                
    
    def run_n_simulations(self, data_path, final_graph_path,  n = 1, store_data = True):
        for n in range(1, n + 1):
            print(f"running simulation {n}...")
            self.run_one_simulation()
            
            if store_data == True:
                self.save_data(path = data_path, simulation_batch_num = n)
                self.save_final_graph_object(path = final_graph_path, simulation_batch_num = n)
                
    def check_stable_state(self):
        if self.I_solution[len(self.I_solution) - 1] == 0 and self.S_solution[len(self.S_solution) - 1] == 0:
            return True
        else:
            return False
                
            
    
    def draw_graph(self, with_labels = True):
        #set position of now network should be plotted each time
        pos = nx.spring_layout(self.G)
        
        color_state_map = {"L": 'blue', "I": 'green', "S": 'yellow', "A": 'red'}
        nx.draw(self.G, node_size = 100, 
                node_color=[color_state_map[node[1]['state']] for node in self.G.nodes(data=True)],
                with_labels = with_labels, pos = pos)
        plt.title('graph')
        plt.show()
        
    def plot_curves(self):
        L_pop_fraction = [element / self.N for element in self.L_solution]
        I_pop_fraction = [element / self.N for element in self.I_solution]
        S_pop_fraction = [element / self.N for element in self.S_solution]
        A_pop_fraction = [element / self.N for element in self.A_solution]
        
        #plot each line with a different label
        plt.plot(L_pop_fraction, label = "Luddite", color = "blue")
        plt.plot(I_pop_fraction, label = "Ignorant", color = "green")
        plt.plot(S_pop_fraction, label = "Succeptible", color = "orange")
        plt.plot(A_pop_fraction, label  = "Accepted", color = "red")
        
        #title = "T-LISA Simulation: " + "N = " + str(N) + ", k = " + str(k) + ", $\gamma{}$ = " + str(round(gamma, 5)) + ", $\omega{}$ = " + str(round(omega, 5)) + " $r$ = " + str(r) + " sim#: " + str(simulation_number) 
        title = "title"
        plt.title(title)
        plt.legend(loc="upper right")
        plt.xlabel("simulations")
        plt.ylabel("Population Proportion")
        
        plt.show()
        
    def save_data(self, path, simulation_batch_num = "not batched!"):
        #This function will throq an error if run_simulation hasn't been run prioir
        #Consider try-except block?
        
        
        sim_list = list(range(1, self.simulations+1))
        
        #save paramaters that the model was run on
        self.params = [[self.N, self.simulations, self.k, self.P, self.inital_conditions, self.r, self.gamma, self.omega]]
        
        ##note, this weird dataxtructure (dictionary nested in a list) is so that 
        self.params = {"N": self.N, 
                        "simulations": self.simulations, 
                        "k": self.k, 
                        #"P": self.P, -P is calculated from k and N, and makes naming worse, so it's excluded
                        "inital_conditions": self.inital_conditions, 
                        "r": self.r, 
                        "gamma": self.gamma, 
                        "omega": self.omega,
                        "Simulation_batch_num": simulation_batch_num}
        #df["paramaters"] = self.params
        
        #df["L_sol"] = self.L_solution
        
        #Set up the dict that will
        self.data = {#"simulation": sim_list, 
                     "L_sol": self.L_solution, 
                     "I_sol": self.I_solution, 
                     "S_sol": self.S_solution, 
                     "A_sol": self.A_solution}
        
        #define df
        df = pd.DataFrame(self.data)
        df["simulation"] = df.index + 1
        df["params"] = pd.Series([self.params])
        
        #Name the datafile
        file_name = str(self.params).replace(":","=").replace("'", "") + ".json"
        file_path = path + file_name
        
        #Save the file
        df.to_json(file_path)
        
        return df
    
    def save_final_graph_object(self, path, simulation_batch_num = "not batched!"):
        
        #Using the same naming comventions from data saving
        file_name = str(self.params).replace(":","=").replace("'", "") + ".json"
        file_path = path + file_name
        
        
        graph_obj = json_graph.node_link_data(self.G)
        with open(file_path, 'w') as f:
            json.dump(graph_obj, f)
            
    def return_G(self):
        return self.G
            



total_simulations = 750

regression_df = pd.DataFrame(columns = ['Node', 'state', 'betweenness', 'degree', 'closeness', 'clustering', 'triangles'])

for simulation in range(total_simulations):
    #initialize the simulation class
    N = 20
    graph = ER_Graph(N, 200, 2, [0,.8,.2,0], .9, .08, .08)
    
    #run one simulation to reach a stable stare
    graph.run_one_simulation(animate = False)
    
    #store the Graphs as G
    #G = graph.return_G()
    G = graph.return_G().subgraph(max(nx.connected_components(graph.return_G()), key=len)).copy()

    
    #Compute various measures - 
    #Betweeness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    centrality_df = pd.DataFrame.from_dict(betweenness_centrality, orient='index', columns=['betweenness'])
    centrality_df.reset_index(inplace=True)
    centrality_df.rename(columns={'index': 'Node'}, inplace=True)
    
    #degree centrality
    degree_centrality = nx.degree_centrality(G)
    degree_df = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['degree'])
    degree_df['degree'] = degree_df['degree']*(N-1)
    degree_df.reset_index(inplace=True)
    degree_df.rename(columns={'index': 'Node'}, inplace=True)
    
    #closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    closeness_df = pd.DataFrame.from_dict(closeness_centrality, orient='index', columns=['closeness'])
    closeness_df.reset_index(inplace=True)
    closeness_df.rename(columns={'index': 'Node'}, inplace=True)
    
    #Eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    eigenvector_df = pd.DataFrame.from_dict(eigenvector_centrality, orient='index', columns=['eigenvector'])
    eigenvector_df.reset_index(inplace=True)
    eigenvector_df.rename(columns={'index': 'Node'}, inplace=True)
    
    #Clustering Coefficient
    clustering_coef= nx.clustering(G)
    clustering_df = pd.DataFrame.from_dict(clustering_coef, orient='index', columns=['clustering'])
    clustering_df.reset_index(inplace=True)
    clustering_df.rename(columns={'index': 'Node'}, inplace=True)
    
    #triangles
    triangles = nx.triangles(G)
    triangles_df = pd.DataFrame.from_dict(triangles, orient='index', columns=['triangles'])
    triangles_df.reset_index(inplace=True)
    triangles_df.rename(columns={'index': 'Node'}, inplace=True)
    
    #eccentricity - this one is not relevant I think. Every node has eccentricity 3 for the 300 ER graph
    #eccentricity = nx.eccentricity(G)
    
    
    
    #state!
    node_states = {node: data['state'] for node, data in G.nodes(data=True)}
    states_df = pd.DataFrame.from_dict(node_states, orient='index', columns=['state'])
    states_df.reset_index(inplace=True)
    states_df.rename(columns={'index': 'Node'}, inplace=True)
    
    
    
    #Merge all relevant info
    df = states_df.merge(centrality_df, on = 'Node').merge(degree_df, on ='Node').merge(closeness_df, on = "Node").merge(clustering_df, on = "Node").merge(triangles_df, on = "Node").merge(eigenvector_df, on = "Node")
    regression_df = pd.concat([regression_df, df])
    
    #Drop rows that are not in final states (mostly Islands I think)
    print(f"{simulation} / {total_simulations} sims complete")


regression_df['A_dummy'] = regression_df['state'].map(lambda x: x == "A")


#%%read old CSVs
#regression_df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/Python Codes & Outputs/Data Analysis/Regressions_data/50,250,30,[0,8,2,0],9,08,08_300sims_3_36_24.csv")
#regression_df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/Python Codes & Outputs/Data Analysis/Regressions_data/300,250,30,[0,8,2,0],9,08,08_50sims_3_36_24.csv")
regression_df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/Python Codes & Outputs/Data Analysis/Regressions_data/50,200,5,[0,8,2,0],9,08,08_300sims_3_37_24.csv")
#regression_df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/Python Codes & Outputs/Data Analysis/Regressions_data/300,250,5,[0,8,2,0],9,08,08_300sims_3_30_24.csv")
#regression_df = pd.read_csv("/Users/markosuchy/Desktop/Honors Thesis/Python Codes & Outputs/Data Analysis/Regressions_data/20,200,2,[0,8,2,0],9,08,08_750sims_3_30_24.csv")

regression_df_cleaned = regression_df[regression_df['state'].isin(['A', 'L'])]

regression_df['A_dummy'] = regression_df['A_dummy'].astype(int)
regression_df_cleaned['A_dummy'] = regression_df_cleaned['A_dummy'].astype(int)
regression_df['triangles'] = regression_df['triangles'].map(lambda x: int(x))

#%%regression 1
X = regression_df[['betweenness', 'degree', 'closeness']]
y = regression_df['A_dummy']

model = sm.Logit(y, X)
results = model.fit(cov_type = 'HC0')
#results = model.fit()

print(results.summary())

#%% regression 2


X = regression_df[['betweenness', 'degree', 'closeness', 'clustering', 'triangles']]
y = regression_df['A_dummy']

model = sm.Logit(y, X)
results = model.fit(cov_type = 'HC0')
#results = model.fit()

print(results.summary())

#%% Save CSV
#regression_df.to_csv("/Users/markosuchy/Desktop/Honors Thesis/Python Codes & Outputs/Data Analysis/Regressions_data/20,200,2,[0,8,2,0],9,08,08_750sims_3_30_24.csv")

#%%regression 1
X = regression_df_cleaned[['betweenness', 'degree', 'closeness']]
y = regression_df_cleaned['A_dummy']

model_1 = sm.Logit(y, X)
results_1 = model_1.fit(cov_type = 'HC0')
#results = model.fit()

print(results.summary())

#%% regression 2
regression_df_cleaned['triangles'] = regression_df_cleaned['triangles'].map(lambda x: int(x))

X = regression_df_cleaned[['betweenness', 'degree', 'closeness', 'clustering', 'triangles']]
y = regression_df_cleaned['A_dummy']

model_2 = sm.Logit(y, X)
results_2 = model_2.fit()
#results = model.fit()

print(results_2.summary())


#%%TRY LPM:
#Convert to LPM
X = regression_df_cleaned[['betweenness', 'degree', 'closeness', 'eigenvector']]
X = X = sm.add_constant(X)

y = regression_df_cleaned['A_dummy']

lpm = sm.OLS(y, X)
lpm_results = lpm.fit()

print(lpm_results.summary())


#%%TRY LPM:
#Convert to LPM
X = regression_df_cleaned[['degree', 'closeness', 'betweenness', 'eigenvector', 'clustering', 'triangles']]
X = X = sm.add_constant(X)

y = regression_df_cleaned['A_dummy']

lpm = sm.OLS(y, X)
lpm_results = lpm.fit()

print(lpm_results.summary())


#%%Noramlize stuff


#Noramlize Everything
regression_df_cleaned['degree_normalized'] = (regression_df_cleaned['degree'] - regression_df_cleaned['degree'].min()) / (regression_df_cleaned['degree'].max() - regression_df_cleaned['degree'].min())
regression_df_cleaned['betweenness_normalized'] =  (regression_df_cleaned['betweenness'] - regression_df_cleaned['betweenness'].min()) / (regression_df_cleaned['betweenness'].max() - regression_df_cleaned['betweenness'].min())
regression_df_cleaned['closeness_normalized'] = (regression_df_cleaned['closeness'] - regression_df_cleaned['closeness'].min()) / (regression_df_cleaned['closeness'].max() - regression_df_cleaned['closeness'].min())
regression_df_cleaned['clustering_normalized'] = (regression_df_cleaned['clustering'] - regression_df_cleaned['clustering'].min()) / (regression_df_cleaned['clustering'].max() - regression_df_cleaned['clustering'].min())
regression_df_cleaned['eigenvector_normalized'] = (regression_df_cleaned['eigenvector'] - regression_df_cleaned['eigenvector'].min()) / (regression_df_cleaned['eigenvector'].max() - regression_df_cleaned['eigenvector'].min())
regression_df_cleaned['triangles_normalized'] = (regression_df_cleaned['triangles'] - regression_df_cleaned['triangles'].min()) / (regression_df_cleaned['triangles'].max() - regression_df_cleaned['triangles'].min())

#%%TRY LPM:
#Convert to LPM

X = regression_df_cleaned[['degree_normalized', 'closeness_normalized', 'betweenness_normalized', 'eigenvector_normalized', 'clustering_normalized', 'triangles_normalized']]
X = sm.add_constant(X)

y = regression_df_cleaned['A_dummy']

lpm = sm.OLS(y, X)
lpm_results = lpm.fit()

print(lpm_results.summary().as_latex())


#%%
plt.figure()
regression_df_cleaned.boxplot(["closeness", "betweenness", "eigenvector", "clustering"])
#plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/close_between_eigen_clust_N=300", dpi = 900)


#%%logit fit test
X = regression_df_cleaned[['closeness_normalized', 'betweenness_normalized', 'eigenvector_normalized', 'clustering_normalized', 'triangles_normalized']]
y = regression_df_cleaned['A_dummy']

model = sm.Logit(y, X)
results = model.fit()
#results = model.fit()

print(results.summary())


