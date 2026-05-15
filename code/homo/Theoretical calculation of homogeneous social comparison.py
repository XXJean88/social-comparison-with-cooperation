import networkx as nx
import numpy as np
import warnings

# Ignore NumPy compatibility warnings
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")

def Coalescence_Times_Homogeneous_Lambda(G, lambda_val):
    """
    Coalescence time calculation function for homogeneous λ
    """
    adjacencyMatrix = nx.adjacency_matrix(G).todense()
    N = len(G.nodes)
    
    # Compute transition probability matrix
    P = np.zeros((N, N))
    for row in range(N):
        sumOverRow = np.sum(adjacencyMatrix[row])
        for col in range(N):
            P[row, col] = adjacencyMatrix[row, col] / sumOverRow
    
    # Compute higher-order transition probabilities
    P2 = P @ P
    P3 = P @ P @ P
    P4 = P @ P @ P @ P
    
    # Compute stationary distribution
    Pi = np.zeros(N)
    W = np.sum(adjacencyMatrix)
    for i in range(N):
        Pi[i] = np.sum(adjacencyMatrix[i]) / W

    # Build linear system
    A = np.zeros((N * N, N * N))
    B = np.ones((N * N, 1)) / 2
    
    # Diagonal elements handling
    eta_ii = [i * N + i for i in range(N)]
    for row in range(N * N):
        if row in eta_ii:
            A[row, row] = 1
            B[row, 0] = 0
        else:
            i = int(row / N)
            j = row % N
            # First block: transitions based on j
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    A[row, k] = 1 - 0.5 * P[j, k % N]
                else:
                    A[row, k] = -0.5 * P[j, k % N]
            # Second block: transitions based on i
            for k in range(j * N, (j + 1) * N):
                A[row, k] = -0.5 * P[i, k % N]

    # Solve linear system
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
    
    Eta = X.reshape(N, N)

    # Compute coalescence time terms
    eta_1 = 0
    eta_2 = 0
    eta_3 = 0
    eta_4 = 0
    
    for i in range(N):
        for j in range(N):
            eta_1 += Pi[i] * P[i, j] * Eta[i, j]
            eta_2 += Pi[i] * P2[i, j] * Eta[i, j]
            eta_3 += Pi[i] * P3[i, j] * Eta[i, j]
            eta_4 += Pi[i] * P4[i, j] * Eta[i, j]
    
    # Compute threshold using the homogeneous λ formula
    numerator = eta_2 + lambda_val * (eta_3 - eta_1)
    denominator = eta_3 - eta_1 + lambda_val * (eta_4 - eta_2)
    
    if abs(denominator) < 1e-12:
        return np.inf
    else:
        threshold = numerator / denominator
        return threshold

# Test code
n = 100
final_seed = 42

# Select network type
graph_type = "ws"  # can be changed to "rg", "ba", or "ws"

if graph_type == "rg":
    G = nx.random_graphs.random_regular_graph(4, n, seed=final_seed)
    while not nx.is_connected(G):
        final_seed = np.random.randint(10000)
        G = nx.random_graphs.random_regular_graph(4, n, seed=final_seed)
    filename_prefix = "rg"
elif graph_type == "ba":
    G = nx.barabasi_albert_graph(n=n, m=2, seed=final_seed)
    while not nx.is_connected(G):
        final_seed = np.random.randint(10000)
        G = nx.barabasi_albert_graph(n=n, m=2, seed=final_seed)
    filename_prefix = "ba"
elif graph_type == "ws":
    G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=final_seed)
    while not nx.is_connected(G):
        final_seed = np.random.randint(10000)
        G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=final_seed)
    filename_prefix = "ws"
    
print(f"Final random seed used: {final_seed}")
print(f"Network type: {graph_type}")
print(f"Number of nodes: {n}")
print(f"Number of edges: {G.number_of_edges()}")

# Generate range of λ values
lambda_values = np.arange(-10, 10, 0.5)
thresholds = []

# Iterate over all λ values
for idx, lambda_val in enumerate(lambda_values):
    threshold = Coalescence_Times_Homogeneous_Lambda(G, lambda_val)
    thresholds.append(threshold)
    
    # Show progress
    status = "∞" if threshold == np.inf else f"{threshold:.6f}"
    print(f"Progress: {idx+1:2d}/{len(lambda_values):2d} | λ = {lambda_val:5.1f} | threshold = {status}")

# Output summary of results
finite_thresholds = [t for t in thresholds if t != np.inf]
finite_lambdas = [lambda_values[i] for i, t in enumerate(thresholds) if t != np.inf]

# Save results to file
results = []
for lambda_val, threshold in zip(lambda_values, thresholds):
    results.append([lambda_val, threshold if threshold != np.inf else float('inf')])

results_array = np.array(results)
filename = f'homogeneous_lambda_{filename_prefix}_{n}.csv'
np.savetxt(filename, results_array, delimiter=',', 
            header='lambda,threshold', comments='', fmt='%.6f')
print(f"Results saved to: {filename}")
