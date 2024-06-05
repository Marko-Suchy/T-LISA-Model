#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:03:17 2024

T-LISA Parameter sweep

This code is meant run simulations across the gamut of parameter space!
We pull the "Simulation Step" function from T-LISA_Model_Simulation.py

Psudocode:
    
    write ER_Graph class, which can run an ER_Graph Simulation and store data
        Note, this function can run simulations in BATCHES, which are multiple runs with the same paramater set 
    write Rescaled_MFT class, which can run the rescaled MFT code and store data
    Loop over paramater space:
        run ER Graph sim
        run rescaled MFT sim
        calculate average difference in results
        create a CSV which has each param as a column, and the average difference for each solution path (L, I, S, A)

@author: markosuchy
"""
#%%
import networkx as nx
import matplotlib.pyplot as plt
import random 
import pandas as pd

import scipy, scipy.integrate
import numpy as np

import json
from networkx.readwrite import json_graph

#%%
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
        self.__init__(self.N, self.simulations, self.k, self.inital_conditions, self.r, self.gamma, self.omega)
        
        
       #make sure solutions lists are empty
        self.L_solution = []
        self.I_solution = []
        self.S_solution = []
        self.A_solution = []
        
       #Run the simulation:
        for simulation in range(self.simulations):
            
            if animate:
                self.draw_graph()
            
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
        #plt.show()
        
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

        
        




class Rescaled_MFT():
    def __init__(self, N, tMax, k, inital_conditions, r, gamma, omega):
        self.N = N
        self.k = k
        self.inital_conditions = inital_conditions
        self.r = r
        self.gamma = gamma
        self.omega = omega
        self.tMax = tMax
            
    # This defines a function that is the right-hand side of the ODEs
    # Warning!  Whitespace at the begining of a line is significant!
    def rhs(self, Y, t, omega, gamma, r):
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
        dL = self.gamma * self.r * S * I + self.omega * self.r * A * I
        #dI = -(S+A)*I - self.gamma * self.r * S * I - self.omega * self.r * A *  - Unscaled version
        dI = -((self.gamma * self.r) + self.k/self.N)*S*I - A*(self.k/self.N)*I - self.omega * self.r * A * I
        #dS = (S+A)*I - self.gamma * S - unscaled version
        dS = S * ((self.k/self.N) * I - self.gamma) + A * (self.k/self.N) * I
        dA = self.gamma*S
        # Convert meaningful component vectors into a single vector
        dY = [ dL, dI, dS, dA ]
    
        return dY
    
    def solve_MFT(self):
        # Time vector for solution
        self.T = np.linspace(0, self.tMax, self.tMax*20 + 1)
        solution = scipy.integrate.odeint(self.rhs,self.inital_conditions, self.T, args = (self.omega, self.gamma, self.r))
                
        self.L_sol = solution[:, 0]
        self.I_sol = solution[:, 1]
        self.S_sol = solution[:, 2]
        self.A_sol = solution[:, 3]
    
    #Note - I don't actually need to save data here bc this isn't stochastic and can be run very quickly.
    #But... maybe I should?
    def concat_data(self):
        #Set up the inital df
        data = {"T" : self.T,
                "L_sol_MFT" : self.L_sol,
                "I_sol_MFT" : self.I_sol,
                "S_sol_MFT" : self.S_sol,
                "A_sol_MFT" : self.A_sol
            }
        self.df = pd.DataFrame(data)
        
        #add params
        self.params = {"N": self.N, 
                        "tMax": self.tMax, 
                        "k": self.k, 
                        #"P": self.P, -P is calculated from k and N, and makes naming worse, so it's excluded
                        "inital_conditions": self.inital_conditions, 
                        "r": self.r, 
                        "gamma": self.gamma, 
                        "omega": self.omega
                        }
        self.df["params"] = pd.Series([self.params])
        
        ##NEED THE ACTUAL PA
        
    
    def plot_curves(self):
        
        plt.plot(self.T, self.L_sol, label = "Luddites", color = "blue")
        plt.plot(self.T, self.I_sol, label = "Ignorants", color = "green")
        plt.plot(self.T, self.S_sol, label = "Susceptibles", color = "orange")
        plt.plot(self.T, self.A_sol, label = "Adopters", color = "red")
        
        plt.show()

#%%Testing MFT class
"""
MFT_1 = Rescaled_MFT(300, 10, 20, [0, .8, .2, 0], 0.9, .5, .5)
MFT_1.solve_MFT()
MFT_1.concat_data()

#df = MFT_1.df
#%%Test ER_Graph class
data_path = "/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/Param_Sweep_Data/"

Graph1 = ER_Graph(100, 100, 20, [0, .8, .2, 0], 0.9, .5, .5) 
Graph1.draw_graph()
#Graph1.run_one_simulation()
#Graph1.run_n_simulations(data_path = data_path, n = 1)
Graph1.plot_curves()

Graph1.run_one_simulation()

Graph1.run_n_simulations(data_path, n = 3, store_data=True)

#df = Graph1.save_data(path = data_path)  """



#%%SWEEEP! 
#Sweep variables
"""
N_space = np.linspace(50, 200, num = 8, dtype=int)
r_space = np.linspace(.5, 1, num = 5)
gamma_space = np.linspace(.1, 1, num = 5)
omega_space = np.linspace(.1, 1, num = 5)
k_space = np.linspace(10, 30, num = 3)  """

N_space = [100]
r_space = [.9]
gamma_space = [.08]
omega_space = [.08]
k_space = [20] 


#Constants
simulations = 100
inital_conditions = [0, .8, .2, 0]
data_path = "/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/Param_Sweep_Data/"
final_graph_path = ""
batch_size = 5



#Parameter Space Setup
'''
N_space = np.linspace(50, 200, num = 8, dtype=int)
r_space = np.linspace(.5, 1, num = 5)
gamma_space = np.linspace(.1, 1, num = 5)
omega_space = np.linspace(.1, 1, num = 5)
k_space = np.linspace(10, 30, num = 3) '''

 
output_csv = pd.DataFrame(columns = ("N", "r", "gamma", "omega", "k", "L_avg_err", "I_avg_err", "S_avg_err", "A_avg_err"))

#Run the sweep!
for N in N_space:
    print(f"N space at {N}")
    for r in r_space:
        print(f"r space at {r}")
        for gamma in gamma_space:
            print(f"gamma space at {gamma}")
            for omega in omega_space:
                print(f"omega space at {omega}")
                for k in k_space:
                    print(f"k space at {k}")
                    print(f"r space at {r}")
                    print(f"N space at {N}")
                    print(f"gamma space at {gamma}")
                    print(f"omega space at {omega}")
                    
                    
                    #RUN THE SIMULATION AND COMPARE STUFF!
                    er_graph = ER_Graph(N, simulations, k, inital_conditions, r, gamma, omega)
                    er_graph.run_n_simulations(data_path, n=batch_size)
                    
                    #save final state
                    final_state = json_graph.node_link_data(er_graph)

                    
                    
                    
                    #Run MFT for the same 
                    MFT = Rescaled_MFT(N, simulations, k, inital_conditions, r, gamma, omega)
                    MFT.solve_MFT()
                    MFT.concat_data()
                    MFT_df = MFT.df
                    
                    #Compare each data peice to MFT
                    for batch in range(1, batch_size+1):
                        params = {"N": N, 
                                  "simulations": simulations, 
                                        "k": k, 
                                        #"P": self.P, -P is calculated from k and N, and makes naming worse, so it's excluded
                                        "inital_conditions": inital_conditions, 
                                        "r": r, 
                                        "gamma": gamma, 
                                        "omega": omega,
                                        "Simulation_batch_num": batch}
                        file_name = str(params).replace(":","=").replace("'", "") + ".json"
                        file_path = data_path + file_name
                        er_graph_df = pd.read_json(file_path)
                        
                        #Convet er graph solutions to population proportions
                        
                        er_graph_df["L_sol"] = er_graph_df["L_sol"] / N
                        er_graph_df["I_sol"] = er_graph_df["I_sol"] / N
                        er_graph_df["S_sol"] = er_graph_df["S_sol"] / N
                        er_graph_df["A_sol"] = er_graph_df["A_sol"] / N
            
                        ##COMPUTE ERROR
                        #Inner join er graph df and MFT df on Time/simulation
                        comparison_df = pd.merge(er_graph_df, MFT_df, how = "inner", left_on=("simulation"), right_on=("T"))
                        
                        #calculate error
                        comparison_df["L_difference"] = comparison_df["L_sol"] - comparison_df["L_sol_MFT"]
                        comparison_df["I_difference"] = comparison_df["I_sol"] - comparison_df["I_sol_MFT"]
                        comparison_df["S_difference"] = comparison_df["S_sol"] - comparison_df["S_sol_MFT"]
                        comparison_df["A_difference"] = comparison_df["A_sol"] - comparison_df["A_sol_MFT"]
                        
                        #compute average error
                        L_avg_error = comparison_df["L_difference"].mean()
                        I_avg_error = comparison_df["I_difference"].mean()
                        S_avg_error = comparison_df["S_difference"].mean()
                        A_avg_error = comparison_df["A_difference"].mean()
                        
                        ##PLOT THE THANG
                        if batch == 1:
                            plt.plot(comparison_df["T"], comparison_df["L_sol_MFT"], color = "blue", label = "Luddite")
                            plt.plot(comparison_df["T"], comparison_df["I_sol_MFT"], color = "green", label = "Ignorant")
                            plt.plot(comparison_df["T"], comparison_df["S_sol_MFT"], color = "orange", label = "Susceptible")
                            plt.plot(comparison_df["T"], comparison_df["A_sol_MFT"], color = "red", label = "Adopted")
                            
                            plt.xlabel("time (t)")
                            plt.ylabel("Population Proportion")
                            
                            plt.legend()
                            plt.title("MFT vs. Agent-Based Simulation")
                            
                        
                        plt.plot(comparison_df["simulation"], comparison_df["L_sol"], color = "blue", alpha = 0.3)
                        plt.plot(comparison_df["simulation"], comparison_df["I_sol"], color = "green", alpha = 0.3)
                        plt.plot(comparison_df["simulation"], comparison_df["S_sol"], color = "orange", alpha = 0.3)
                        plt.plot(comparison_df["simulation"], comparison_df["A_sol"], color = "red", alpha = 0.3)
                        
                        
                        ##ADD TO OUTPUT CSV 
                        #Output a row of CSV Data
                        output_csv.loc[len(output_csv)] = [N, r, gamma, omega, k, L_avg_error, I_avg_error, S_avg_error, A_avg_error]
                        
                        
                        
                        
                        
                        print(f"r space at {r}")

output_csv['avg_err'] = (abs(output_csv['L_avg_err']) + abs(output_csv["I_avg_err"]) + abs(output_csv["S_avg_err"]) + abs(output_csv["A_avg_err"])) / 4


#plt.text(0, 100, str(comparison_df["params_x"][0]))
#plt.show()
#plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/MFT_VS_AGENT_2_20_24_2.png", dpi = 900)
                    
#%%Make Plots with MFT and Stochastic runs



simulations = 250
inital_conditions = [0, .8, .2, 0]
data_path = "/Users/markosuchy/Desktop/Honors Thesis/json data/Erdos_Renyi/3_22_24/Data/"
batch_size = 5

#plot both of them on the same plot
#batch_size = 2

N = 92
r = 0.875
gamma = 0.325
omega = 0.325
k = 20.0

#Run MFT for the same 
MFT = Rescaled_MFT(N, simulations, k, inital_conditions, r, gamma, omega)
MFT.solve_MFT()
MFT.concat_data()
MFT_df = MFT.df


L_batch_avg_df = pd.DataFrame()
I_batch_avg_df = pd.DataFrame()
S_batch_avg_df = pd.DataFrame()
A_batch_avg_df = pd.DataFrame()

for batch in range(1, batch_size+1):
    params = {"N": N, 
              "simulations": simulations, 
                    "k": k, 
                    #"P": self.P, -P is calculated from k and N, and makes naming worse, so it's excluded
                    "inital_conditions": inital_conditions, 
                    "r": r, 
                    "gamma": gamma, 
                    "omega": omega,
                    "Simulation_batch_num": batch}
    file_name = str(params).replace(":","=").replace("'", "") + ".json"
    file_path = data_path + file_name
    er_graph_df = pd.read_json(file_path)
    
    rows_to_add = 40 - len(er_graph_df)
    if rows_to_add > 0:
        row_to_duplicate = er_graph_df.iloc[len(er_graph_df) - 1: len(er_graph_df)]
        duplicated_rows = pd.concat([row_to_duplicate]*rows_to_add, ignore_index=True)
        er_graph_df = pd.concat([er_graph_df, duplicated_rows], ignore_index=True)
        
    er_graph_df["simulation"] = er_graph_df.index + 1

        
    #Convet er graph solutions to population proportions
    
    er_graph_df["L_sol"] = er_graph_df["L_sol"] / N
    er_graph_df["I_sol"] = er_graph_df["I_sol"] / N
    er_graph_df["S_sol"] = er_graph_df["S_sol"] / N
    er_graph_df["A_sol"] = er_graph_df["A_sol"] / N

    ##COMPUTE ERROR
    #Inner join er graph df and MFT df on Time/simulation
    comparison_df = pd.merge(er_graph_df, MFT_df, how = "left", left_on=("simulation"), right_on=("T"))
    
    #calculate error
    comparison_df["L_difference"] = comparison_df["L_sol"] - comparison_df["L_sol_MFT"]
    comparison_df["I_difference"] = comparison_df["I_sol"] - comparison_df["I_sol_MFT"]
    comparison_df["S_difference"] = comparison_df["S_sol"] - comparison_df["S_sol_MFT"]
    comparison_df["A_difference"] = comparison_df["A_sol"] - comparison_df["A_sol_MFT"]
    
    #compute average error
    L_avg_error = comparison_df["L_difference"].mean()
    I_avg_error = comparison_df["I_difference"].mean()
    S_avg_error = comparison_df["S_difference"].mean()
    A_avg_error = comparison_df["A_difference"].mean()
    
    ##PLOT THE THANG
    plt.plot(comparison_df["simulation"], comparison_df["L_sol"], color = "blue", alpha = 0.3)
    plt.plot(comparison_df["simulation"], comparison_df["I_sol"], color = "green", alpha = 0.3)
    plt.plot(comparison_df["simulation"], comparison_df["S_sol"], color = "orange", alpha = 0.3)
    plt.plot(comparison_df["simulation"], comparison_df["A_sol"], color = "red", alpha = 0.3)
    
    font = {"size": 11}
    #plt.title(str(comparison_df["params_x"][0]["N"]) + str(comparison_df["params_x"][0]["simulations"]) + str(comparison_df["params_x"][0]["k"]) + str(comparison_df["params_x"][0]["inital_conditions"]) + str(comparison_df["params_x"][0]["r"]) + str(comparison_df["params_x"][0]["gamma"]) + str(comparison_df["params_x"][0]["omega"]), fontdict=font)
    #plt.text(str(comparison_df["params_x"][0]))
    
    #setup df for batch avg
    L_batch_avg_df[f"L_sol_{batch}"] = er_graph_df["L_sol"]
    I_batch_avg_df[f"I_sol_{batch}"] = er_graph_df["I_sol"]
    S_batch_avg_df[f"S_sol_{batch}"] = er_graph_df["S_sol"]
    A_batch_avg_df[f"A_sol_{batch}"] = er_graph_df["A_sol"]


plt.plot(comparison_df["T"], comparison_df["L_sol_MFT"], color = "blue")
plt.plot(comparison_df["T"], comparison_df["I_sol_MFT"], color = "green")
plt.plot(comparison_df["T"], comparison_df["S_sol_MFT"], color = "orange")
plt.plot(comparison_df["T"], comparison_df["A_sol_MFT"], color = "red")

#Compute batch avgs (this only works proberly for batch size 5!)
L_batch_avg_df['avg'] = L_batch_avg_df.mean(axis = 1)
I_batch_avg_df['avg'] = I_batch_avg_df.mean(axis = 1)
S_batch_avg_df['avg'] = S_batch_avg_df.mean(axis = 1)
A_batch_avg_df['avg'] = A_batch_avg_df.mean(axis = 1)

plt.plot(L_batch_avg_df.index + 1,  L_batch_avg_df['avg'], linestyle = "dashed", color = "blue", alpha = 0.6)
plt.plot(I_batch_avg_df.index + 1,  I_batch_avg_df['avg'], linestyle = "dashed", color = "green", alpha = 0.6)
plt.plot(S_batch_avg_df.index + 1,  S_batch_avg_df['avg'], linestyle = "dashed", color = "orange", alpha = 0.6)
plt.plot(A_batch_avg_df.index + 1,  A_batch_avg_df['avg'], linestyle = "dashed", color = "red", alpha = 0.6)


title = "T-LISA MC vs MFT: " + "N = " + str(N) + ", k = " + str(k) + ", $\gamma{}$ = " + str(round(gamma, 5)) + ", $\omega{}$ = " + str(round(omega, 5)) + " $r$ = " + str(r) 
plt.title(title)

plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/T-LISA_Stochastic_and_MFT/" + title + ".png", dpi = 900)

#%%

G = ER_Graph(50, 150, 30, [0, 0.8, 0.2, 0], 0.9, .08, .08)
#plt.figure()
#plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/N_50_init_graph.png", dpi= 900)

G.run_one_simulation()

G.draw_graph()
plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/N_50_k=30_stable_graph.png", dpi= 900)



plt.clf()
G2 = ER_Graph(50, 150, 10, [0, 0.8, 0.2, 0], 0.9, .08, .08)
G2.run_one_simulation()
G2.draw_graph()
plt.savefig("/Users/markosuchy/Desktop/Honors Thesis/Figures and Visualizations/figures/N_50_k=10_stable_graph.png", dpi= 900)

        
