import networkx as nx
import random
import numpy as np
from numba import jit
import multiprocessing
import functools
import math
import warnings
warnings.filterwarnings('ignore')

# Read lambda values from file
def load_lambda_values(filename):
    """Read lambda value for each node from file"""
    lambda_values = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # skip comment lines and empty lines
                parts = line.split()
                if len(parts) >= 2:
                    node_idx = int(parts[0])
                    lambda_val = float(parts[1])
                    # Ensure lambda values are stored in node order
                    while len(lambda_values) <= node_idx:
                        lambda_values.append(0.0)
                    lambda_values[node_idx] = lambda_val
    return np.array(lambda_values)

@jit(nopython=True)  # accelerate with Numba
def single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array):
    """Each node randomly selects one neighbor to play the game"""
    for i in range(len(state_array)):  # iterate over all nodes
        # Randomly select a neighbor
        nbrs_num = deg_array[i]  # number of neighbors of node i
        nbrs_array = nbr_mat[i][:nbrs_num]  # list of neighbors
        
        random_idx = random.randint(0, nbrs_num - 1)  # randomly choose neighbor index
        random_neighbor = nbrs_array[random_idx]  # get the random neighbor
        
        # Play game only with the randomly selected neighbor
        payoff_array[i] = game_matrix[state_array[i]][state_array[random_neighbor]]
    return payoff_array  # return payoff array

@jit(nopython=True)
def calculate_fitness(payoff_array, nbr_mat, deg_array, node_idx, w, lambda_array):
    """
    Calculate fitness: F_i(x) = exp(w * (u_i(x) + λ_i * sum(p_im * u_m(x))))
    Heterogeneous λ: each node uses its own lambda parameter
    """
    # Node's own payoff
    self_payoff = payoff_array[node_idx]
    
    # Get neighbors
    nbrs_num = deg_array[node_idx]  # number of neighbors
    nbrs_array = nbr_mat[node_idx][:nbrs_num]  # list of neighbors
    
    # Calculate average payoff of neighbors
    neighbors_avg_payoff = np.sum(payoff_array[nbrs_array]) / nbrs_num if nbrs_num > 0 else 0
    
    # Get lambda value for this node
    lambda_param = lambda_array[node_idx]
    
    # Calculate fitness (including neighbor payoff)
    fitness = math.exp(w * (self_payoff + lambda_param * neighbors_avg_payoff))
    
    return fitness

@jit(nopython=True)
def strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_array):
    """Strategy update"""
    # Randomly select an individual to update (all individuals update at same rate)
    update_node = random.randint(0, nodesnum - 1)
    
    # Get neighbor information
    nbrs_num = deg_array[update_node]  # number of neighbors
    nbrs_array = nbr_mat[update_node][:nbrs_num]  # list of neighbors
    
    # Calculate fitness of all neighbors
    fitness_array = np.zeros(nbrs_num)  # initialize fitness array
    for i in range(nbrs_num):
        neighbor_idx = nbrs_array[i]  # neighbor index
        fitness_array[i] = calculate_fitness(payoff_array, nbr_mat, deg_array, neighbor_idx, w, lambda_array)
    
    # Select strategy based on fitness
    total_fitness = np.sum(fitness_array)  # total fitness
    if total_fitness > 0:
        prob_array = fitness_array / total_fitness  # selection probabilities
        chosen_neighbor = rand_pick_list(nbrs_array, prob_array)  # pick neighbor according to probability
        state_array[update_node] = state_array[chosen_neighbor]  # update strategy
    
    return state_array

@jit(nopython=True)
def evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_array):
    total_generation = int(1e9)  # total generations
    payoff_array = np.zeros(nodesnum, dtype=np.float64)  # payoff array
    state_array = np.zeros(nodesnum, dtype=np.int_)  # state array (0: defector, 1: cooperator)
    coop_ini = np.random.choice(nodesnum)  # randomly select an initial cooperator
    state_array[coop_ini] = 1  # set initial cooperator
    
    for time in range(total_generation):
        # Single round of game and strategy update
        payoff_array = single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array)
        state_array = strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_array)
        payoff_array[:] = 0  # reset payoff array
        
        coord = np.sum(state_array)  # number of cooperators
        
        # Check if absorbing state is reached
        if coord > nodesnum - 1:  # all nodes cooperate
            return 1
        if coord == 0:  # all nodes defect
            return 0
            
    return coord / nodesnum  # return cooperation proportion

@jit(nopython=True)
def process(core, b, nbr_mat, deg_array, nodesnum, lambda_array):
    w = 0.01  # selection strength parameter
    # Define payoff matrix (prisoner's dilemma)
    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0      # mutual defection
    game_matrix[0][1] = b      # self defect, opponent cooperate
    game_matrix[1][0] = -1     # self cooperate, opponent defect
    game_matrix[1][1] = b - 1  # mutual cooperation
    
    repeat_time = int(1e6)  # number of repetitions
    repeat_array = np.zeros(repeat_time)  # result array
    
    for rep in range(repeat_time):
        # Run evolution process
        freq_c = evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_array)
        repeat_array[rep] = freq_c
    
    # Count cooperation fixations and total absorptions
    coop_fixations = np.sum(repeat_array == 1)
    total_absorptions = np.sum(repeat_array == 1) + np.sum(repeat_array == 0)
    
    return coop_fixations, total_absorptions

@jit(nopython=True)
def rand_pick_list(pick_list, prob_list):
    """Select an element from a list according to probability distribution"""
    x = random.uniform(0, 1)  # generate random number
    cumulative_probability = 0.0  # cumulative probability
    for item, item_probability in zip(pick_list, prob_list):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
    return item

def nbr_dict_mat(nbr_dict):
    """Convert neighbor dictionary to numpy array"""
    nodesnum = len(nbr_dict)  # number of nodes
    nbr_mat = np.zeros([nodesnum, nodesnum], int)  # neighbor matrix
    deg_array = np.zeros(nodesnum, int)  # degree array
    
    for i, nbrs in nbr_dict.items():
        deg_array[i] = len(nbrs)  # record node degree
        if len(nbrs) > 0:
            nbr_mat[i][:len(nbrs)] = np.array(nbrs)  # fill neighbor information
            
    return nbr_mat, deg_array

def compute_rho_c_for_b_value(b, nbr_mat, deg_array, nodesnum, lambda_array, cache_dict):
    """Compute rho_c for a single b value, using cache"""
    # Check cache
    if b in cache_dict:
        return cache_dict[b]
    
    cpu_cores_num = 7  # number of CPU cores
    core_list = np.arange(cpu_cores_num)  # core list
    
    # Use partial to fix parameters
    pt = functools.partial(process, b=b, nbr_mat=nbr_mat, deg_array=deg_array,
                           nodesnum=nodesnum, lambda_array=lambda_array)
    
    # Parallel computation with multiprocessing
    with multiprocessing.Pool(cpu_cores_num) as pool:
        results = pool.map(pt, core_list)
    
    # Aggregate results from all processes
    total_coop_fixations = sum(result[0] for result in results)
    total_absorptions = sum(result[1] for result in results)
    
    # Compute cooperation fixation probability
    rho_c = total_coop_fixations / total_absorptions if total_absorptions > 0 else 0
    
    # Store in cache
    cache_dict[b] = rho_c
    
    return rho_c

def find_b_for_rho_c_0_2(lambda_filename, network_name="BA"):
    """Find b value that makes rho_c closest to 0.2"""
    print(f"Processing lambda distribution: {lambda_filename}")
    
    # Generate graph with n=50
    n = 50
    # G = nx.barabasi_albert_graph(n=n, m=2, seed=42)
    # G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=42)
    G = nx.random_graphs.random_regular_graph(4, n, seed=42)
    
    # Extract network information
    nbrs_dict = nx.to_dict_of_lists(G)  # neighbor dictionary
    nbr_mat, deg_array = nbr_dict_mat(nbrs_dict)  # convert to matrix form
    nodesnum = G.number_of_nodes()  # number of nodes

    # Load lambda values for each node
    lambda_array = load_lambda_values(lambda_filename)
    
    # Verify that lambda array length matches number of nodes
    if len(lambda_array) != nodesnum:
        print(f"Warning: number of lambda values ({len(lambda_array)}) does not match number of nodes ({nodesnum})!")
        if len(lambda_array) < nodesnum:
            lambda_array = np.concatenate([lambda_array, np.zeros(nodesnum - len(lambda_array))])
        else:
            lambda_array = lambda_array[:nodesnum]
    
    print(f"Loaded lambda values for {len(lambda_array)} nodes")
    
    # Set b value range: 5.35 to 5.45, step 0.05
    b_start = 5.35
    b_end = 5.45
    b_step = 0.05                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    
    # Generate array of b values
    b_values = np.arange(b_start, b_end + b_step/2, b_step)
    
    # Compute rho_c for each b value
    cache_dict = {}
    best_b = None
    best_rho_c = None
    min_diff = float('inf')
    
    print(f"Computing b values from {b_start} to {b_end}, step {b_step}")
    
    for b in b_values:
        rho_c = compute_rho_c_for_b_value(b, nbr_mat, deg_array, nodesnum, lambda_array, cache_dict)
        diff = abs(rho_c - 0.02)
        
        print(f"  b={b:.4f}: rho_c={rho_c:.6f}, difference from 0.02 = {diff:.6f}")
        
        if diff < min_diff:
            min_diff = diff
            best_b = b
            best_rho_c = rho_c
    
    # If there are two points straddling 0.2, perform a more precise search
    b_below = None
    b_above = None
    rho_c_below = None
    rho_c_above = None
    
    for b in sorted(cache_dict.keys()):
        rho_c = cache_dict[b]
        if rho_c < 0.2 and (b_below is None or b > b_below):
            b_below = b
            rho_c_below = rho_c
        elif rho_c > 0.2 and (b_above is None or b < b_above):
            b_above = b
            rho_c_above = rho_c
    
    # If the target 0.2 is crossed between two b values, use linear interpolation
    if b_below is not None and b_above is not None:
        print(f"\nTarget 0.2 crossed between b={b_below:.4f} and b={b_above:.4f}")
        print(f"  b={b_below:.4f}: rho_c={rho_c_below:.6f}")
        print(f"  b={b_above:.4f}: rho_c={rho_c_above:.6f}")
        
        # Linear interpolation
        t = (0.2 - rho_c_below) / (rho_c_above - rho_c_below)
        interpolated_b = b_below + t * (b_above - b_below)
        print(f"  Interpolated b value: {interpolated_b:.6f}")
        
        # Compute rho_c at interpolated point to verify
        interpolated_rho_c = compute_rho_c_for_b_value(interpolated_b, nbr_mat, deg_array, nodesnum, lambda_array, cache_dict)
        print(f"  rho_c at interpolated b={interpolated_b:.6f}: {interpolated_rho_c:.6f}")
        
        # Update best values
        interpolated_diff = abs(interpolated_rho_c - 0.2)
        if interpolated_diff < min_diff:
            best_b = interpolated_b
            best_rho_c = interpolated_rho_c
            min_diff = interpolated_diff
    
    print(f"\n{'='*60}")
    print(f"Final result:")
    print(f"b value that makes rho_c closest to 0.2: {best_b:.6f}")
    print(f"Corresponding rho_c: {best_rho_c:.6f}")
    print(f"Difference from 0.2: {min_diff:.6f}")
    print(f"{'='*60}\n")
    
    return best_b

if __name__ == "__main__":
    # List of lambda distribution files
    lambda_files = [
        # "lambda_powerlaw_rg_n50.txt",
        # "lambda_exponential_rg_n50.txt",
        "lambda_uniform_rg_n50.txt",
        # "lambda_normal_rg_n50.txt"
    ]
    
    # Run computation for each lambda distribution
    results = {}
    for lambda_file in lambda_files:
        try:
            print(f"\n{'='*80}")
            print(f"Starting processing: {lambda_file}")
            b_value = find_b_for_rho_c_0_2(lambda_file)
            results[lambda_file] = b_value
        except Exception as e:
            print(f"Error processing file {lambda_file}: {str(e)}")
    
    # Output all results
    print(f"\n{'='*80}")
    print("Summary of results for all lambda distributions:")
    for lambda_file, b_value in results.items():
        dist_name = lambda_file.replace("lambda_", "").replace(".txt", "").replace("_ba_n50", "")
        print(f"{dist_name}: b = {b_value:.6f}")
    print(f"{'='*80}")
