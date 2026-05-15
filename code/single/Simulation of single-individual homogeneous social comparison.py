import networkx as nx
import random
import numpy as np
from numba import jit
import multiprocessing
import functools
import math
import pandas as pd

@jit(nopython=True)  # Accelerate with Numba
def single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array):
    """Each node randomly selects one neighbor to play with"""
    for i in range(len(state_array)):  # Iterate over all nodes
        # Randomly select a neighbor
        nbrs_num = deg_array[i]  # Number of neighbors of node i
        nbrs_array = nbr_mat[i][:nbrs_num]  # Get neighbor list
        
        random_idx = random.randint(0, nbrs_num - 1)  # Random neighbor index
        random_neighbor = nbrs_array[random_idx]  # Get random neighbor
        
        # Play only with the randomly chosen neighbor
        payoff_array[i] = game_matrix[state_array[i]][state_array[random_neighbor]]
    return payoff_array  # Return payoff array

@jit(nopython=True)
def calculate_fitness(payoff_array, nbr_mat, deg_array, node_idx, w, lambda_param, use_neighbors):
    """
    Calculate fitness:
    If use_neighbors=True: F_i(x) = exp(w * (u_i(x) + λ * sum(p_im * u_m(x))))
    If use_neighbors=False: F_i(x) = exp(w * u_i(x))
    """
    # Node's own payoff
    self_payoff = payoff_array[node_idx]
    
    if use_neighbors:
        # Get neighbors
        nbrs_num = deg_array[node_idx]  # Number of neighbors
        nbrs_array = nbr_mat[node_idx][:nbrs_num]  # Neighbor list
        
        # Average payoff of neighbors
        neighbors_avg_payoff = np.sum(payoff_array[nbrs_array]) / nbrs_num
        
        # Calculate fitness (including neighbor payoffs)
        fitness = math.exp(w * (self_payoff + lambda_param * neighbors_avg_payoff))
    else:
        # Calculate fitness (only own payoff)
        fitness = math.exp(w * self_payoff)
    
    return fitness

@jit(nopython=True)
def strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node):
    """Strategy update"""
    # Randomly select one individual to update (all individuals update at the same rate)
    update_node = random.randint(0, nodesnum - 1)
    
    # Get neighbor information
    nbrs_num = deg_array[update_node]  # Number of neighbors
    nbrs_array = nbr_mat[update_node][:nbrs_num]  # Neighbor list
    
    # Calculate fitness for all neighbors
    fitness_array = np.zeros(nbrs_num)  # Initialize fitness array
    for i in range(nbrs_num):
        neighbor_idx = nbrs_array[i]  # Neighbor index
        
        # Check whether this neighbor is the special node (fixed to index 1)
        use_neighbors = (neighbor_idx == special_node)
        
        fitness_array[i] = calculate_fitness(payoff_array, nbr_mat, deg_array, neighbor_idx, w, lambda_param, use_neighbors)
    
    # Choose strategy based on fitness
    total_fitness = np.sum(fitness_array)  # Total fitness
    if total_fitness > 0:
        prob_array = fitness_array / total_fitness  # Selection probabilities
        chosen_neighbor = rand_pick_list(nbrs_array, prob_array)  # Choose neighbor according to probabilities
        state_array[update_node] = state_array[chosen_neighbor]  # Update strategy
    
    return state_array

@jit(nopython=True)
def evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node):
    total_generation = int(1e9)  # Total generations
    payoff_array = np.zeros(nodesnum, dtype=np.float64)  # Payoff array
    state_array = np.zeros(nodesnum, dtype=np.int_)  # State array (0: defect, 1: cooperate)
    coop_ini = np.random.choice(nodesnum)  # Randomly choose initial cooperator
    state_array[coop_ini] = 1  # Set initial cooperator
    
    for time in range(total_generation):
        # Single round of game and strategy update
        payoff_array = single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array)
        state_array = strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node)
        payoff_array[:] = 0  # Reset payoff array
        
        coord = np.sum(state_array)  # Number of cooperators
        
        # Check if absorbing state is reached
        if coord > nodesnum - 1:  # All nodes cooperate
            return 1
        if coord == 0:  # All nodes defect
            return 0
            
    return coord / nodesnum  # Return cooperation fraction

@jit(nopython=True)
def process(core, b, nbr_mat, deg_array, nodesnum, lambda_param, special_node):
    w = 0.01  # Selection intensity parameter
    # Define payoff matrix (Prisoner's Dilemma)
    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0      # Both defect
    game_matrix[0][1] = b      # Self defect, opponent cooperate
    game_matrix[1][0] = -1     # Self cooperate, opponent defect  
    game_matrix[1][1] = b - 1  # Both cooperate
    
    repeat_time = int(1e6)  # Number of repetitions
    repeat_array = np.zeros(repeat_time)  # Result array
    
    for rep in range(repeat_time):
        # Run evolution process
        freq_c = evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node)
        repeat_array[rep] = freq_c
    
    # Count cooperation fixations and total absorptions
    coop_fixations = np.sum(repeat_array == 1)
    total_absorptions = np.sum(repeat_array == 1) + np.sum(repeat_array == 0)
    
    return coop_fixations, total_absorptions

@jit(nopython=True)
def rand_pick_list(pick_list, prob_list):
    """Select an element from the list according to the probability distribution"""
    x = random.uniform(0, 1)  # Random number
    cumulative_probability = 0.0  # Cumulative probability
    for item, item_probability in zip(pick_list, prob_list):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
    return item

def edge_list_array(edge_list):
    """Convert edge list to numpy array"""
    edge_mat = np.zeros([len(edge_list), 2], int)  # Create edge matrix
    for i in range(len(edge_list)):
        edge_mat[i, :] = np.array(edge_list[i])  # Fill edge data
    return edge_mat

def nbr_dict_mat(nbr_dict):
    """Convert neighbor dictionary to numpy array"""
    nodesnum = len(nbr_dict)  # Number of nodes
    nbr_mat = np.zeros([nodesnum, nodesnum], int)  # Neighbor matrix
    deg_array = np.zeros(nodesnum, int)  # Degree array
    
    for i, nbrs in nbr_dict.items():
        deg_array[i] = len(nbrs)  # Record node degree
        if len(nbrs) > 0:
            nbr_mat[i][:len(nbrs)] = np.array(nbrs)  # Fill neighbor information
            
    return nbr_mat, deg_array

if __name__ == "__main__":
    # Generate random regular graph with n=100 and degree=4
    n = 50
    # G = nx.random_graphs.random_regular_graph(4, n, seed=42)
    # G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=42)
    G = nx.barabasi_albert_graph(n=n, m=2, seed=42)

            
    # Extract network information
    edge_list = list(G.edges())  # Edge list
    nbrs_dict = nx.to_dict_of_lists(G)  # Neighbor dictionary
    nbr_mat, deg_array = nbr_dict_mat(nbrs_dict)  # Convert to matrix form
    nodesnum = G.number_of_nodes()  # Number of nodes

    # Define lambda parameter (influence of neighbor payoffs)
    lambda_param = 5
    special_node = 1

    # Compute fixation probability for different b values
    b_array = [6.5]  # Values of payoff parameter b
    cpu_cores_num = 6  # Number of CPU cores
    rhoc_array = []  # Store results
    detailed_results = []  # Store detailed results
    
    for b_para in b_array:
        core_list = np.arange(cpu_cores_num)  # List of cores
        pool = multiprocessing.Pool()  # Create process pool

        # Use partial to fix parameters, including the special node
        pt = functools.partial(process, b=b_para, nbr_mat=nbr_mat, deg_array=deg_array,
                               nodesnum=nodesnum, lambda_param=lambda_param, special_node=special_node)
        results = pool.map(pt, core_list)  # Parallel computation, now returns list of tuples

        # Aggregate results from all processes
        total_coop_fixations = sum(result[0] for result in results)
        total_absorptions = sum(result[1] for result in results)
        
        # Compute cooperation fixation probability
        rho_c = total_coop_fixations / total_absorptions if total_absorptions > 0 else 0
        rhoc_array.append(rho_c)  # Save result
        
        # Save detailed results
        detailed_results.append({
            "b": b_para,
            "coop_fixations": total_coop_fixations,
            "total_absorptions": total_absorptions,
            "rho_c": rho_c
        })
        
        pool.close()  # Close process pool
        pool.join()  # Wait for all processes to finish
        
        print(f"b={b_para}, coop_fixations={total_coop_fixations}, total_absorptions={total_absorptions}, rho_c={rho_c}")

    # Save results as CSV
    result_df = pd.DataFrame({
        "b_value": b_array,
        "cooperation_fixation_probability_rho_c": rhoc_array
    })
    csv_filename = f"ba_n{n}_lambda{lambda_param}_single_neighbor_special_node_1.csv"
    result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_filename = f"detailed_ba_n{n}_lambda{lambda_param}_single_neighbor_special_node.csv"
    detailed_df.to_csv(detailed_csv_filename, index=False, encoding="utf-8-sig")
    
    print(f"Brief results saved to: {csv_filename}")
    print(f"Detailed results saved to: {detailed_csv_filename}")
