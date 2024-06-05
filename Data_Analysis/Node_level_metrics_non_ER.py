#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 23:29:45 2024

@author: markosuchy

This script is meant to run node-level structural attribute regressions on non-ER random graphs
This includes a new Class, "Non_ER_Simulation"
"""


import pandas as pd

import json
from networkx.readwrite import json_graph

import os

import networkx as nx

import statsmodels.api as sm

from matplotlib import pyplot as plt

import random





#%%Define class
class Non_ER_Simulation():
    def __init__(self, G, simulations, state_dict, r, gamma, omega):
        #Initialize global variables
        self.G = G
        self.N = len(G)
        self.k = sum(dict(G.degree()).values()) / len(G.nodes()) #
        self.gamma = gamma
        self.omega = omega
        self.r = r
        self.simulations = simulations
        self.state_dict = state_dict
        
        #Iitialize a bunch of lists that will be plotted 
        self.L_solution = []
        self.I_solution = []
        self.S_solution = []
        self.A_solution = []
        
        #initialize a flags list
        self.breakdown_conditon_flags = []
        
        P = self.k/(self.N-1)
        self.P = P
        
        #Initilize the state
        #state_list = ['L', 'I', 'S', 'A']
        #initial_state = random.choices(state_list, weights=inital_conditions, k = self.N)
        
        nx.set_node_attributes(self.G, self.state_dict, 'state')
    
        
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
            if (self.gamma * len(suceptible_neighbors) + self.omega * len(adopted_neighbors))*(1/len(neighbors)) + (len(suceptible_neighbors) + len(adopted_neighbors))*(1/self.N) >= 1:
                self.breakdown_conditon_flags.append(1)
            
            
            
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
        self.__init__(self.G, self.simulations, self.state_dict, self.r, self.gamma, self.omega)
        
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
            
            if animate:
                self.draw_graph()
            
            if self.check_stable_state():
                print(f"STABLE STATE REACHED at step {simulation + 1}")
                break   
            

                
    
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


#%%Open up some classic graph Objects
G = nx.karate_club_graph()

nx.draw(G, node_size = 100, with_labels = False)
plt.title('graph')
plt.show()


#%%Try to read .edges file
edges_file_path = "/Users/markosuchy/Desktop/Honors Thesis/Network Data/soc-highschool-moreno/soc-highschool-moreno.edges"

'''
# Open the .edges file and skip lines starting with '%'
with open(edges_file_path, 'r') as file:
    for line in file:
        if not line.startswith('%'):  # Skip comment lines
            # Split the line into parts and add an edge to the graph
            node1, node2, weight = line.strip().split()
            G.add_edge(node1, node2, weight=float(weight))

nx.draw(G, node_size = 100, with_labels = False)
'''


#%% write initial conditions dictionary
state_list = ['L', 'I', 'S', 'A']
inital_conditions = [0, 0.2, .8, 0]

initial_state = random.choices(state_list, weights=inital_conditions, k = len(G))
state_dict = dict(zip(G.nodes(), initial_state))

#OR use a manually defined state dict
state_dict = {i: "S" if i == 0 else "I" for i in range(0, len(G))}


#%%

G1 = Non_ER_Simulation(nx.karate_club_graph(), 200, state_dict, 0.9, .08, .08)
G1.run_one_simulation(animate = False)
G1.draw_graph()

G2 = Non_ER_Simulation(nx.karate_club_graph(), 200, state_dict, 0.9, .08, .08)
G2.run_one_simulation(animate = False)
G2.draw_graph()

G3 = Non_ER_Simulation(nx.karate_club_graph(), 200, state_dict, 0.9, .08, .08)
G3.run_one_simulation(animate = False)
G3.draw_graph()

G4 = Non_ER_Simulation(nx.karate_club_graph(), 200, state_dict, 0.9, .08, .08)
G4.run_one_simulation(animate = False)
G4.draw_graph()



#%%Prep Regression DF
regression_df = pd.DataFrame(columns = ['Node', 'state', 'betweenness', 'degree', 'closeness', 'clustering', 'triangles'])

total_simulations = 100

for simulation in range(total_simulations):
    #initilize state dict
    #initial_state = random.choices(state_list, weights=inital_conditions, k = len(G))
    #state_dict = dict(zip(G.nodes(), initial_state))
    
    #initialize the simulation class
    graph = Non_ER_Simulation(G, 150, state_dict, .9, .08, .08)
    
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
    degree_df['degree'] = degree_df['degree']*(len(G)-1)
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


#%%Clean Regression df data
regression_df['A_dummy'] = regression_df['A_dummy'].astype(int)

regression_df['triangles'] = regression_df['triangles'].map(lambda x: int(x))


#%%Run Regressions
X = regression_df[['betweenness', 'degree', 'closeness']]
y = regression_df['A_dummy']

model = sm.Logit(y, X)
results = model.fit(cov_type = 'HC0')
#results = model.fit()

print(results.summary())

#%%
X = regression_df[['degree', 'closeness', 'betweenness', 'eigenvector', 'clustering', 'triangles']]
y = regression_df['A_dummy']

model = sm.Logit(y, X)
results = model.fit(cov_type = 'HC0')
#results = model.fit()

print(results.summary())


#%%TRY LPM:
#Convert to LPM
X = regression_df[['degree', 'closeness', 'betweenness', 'eigenvector', 'clustering', 'triangles']]
X = X = sm.add_constant(X)

y = regression_df['A_dummy']

lpm = sm.OLS(y, X)
lpm_results = lpm.fit()

print(lpm_results.summary())
    
    