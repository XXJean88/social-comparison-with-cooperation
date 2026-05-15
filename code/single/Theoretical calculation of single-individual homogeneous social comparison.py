import networkx as nx
import numpy as np
import warnings
import pickle

# Ignore NumPy compatibility warnings
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")


def precompute_network_constants(G, lambda_node):
    """
    Precompute network constants independent of λ, and the required linear coefficients.
    """
    # Convert adjacency matrix to ndarray to avoid odd behavior of matrix
    adjacencyMatrix = np.asarray(nx.adjacency_matrix(G).todense())
    N = len(G.nodes)

    # Compute transition probability matrix P
    P = np.zeros((N, N))
    for row in range(N):
        sumOverRow = np.sum(adjacencyMatrix[row])
        if sumOverRow > 0:
            P[row, :] = adjacencyMatrix[row] / sumOverRow

    # Higher-order transition probabilities
    P2 = P @ P
    P3 = P2 @ P

    # Stationary distribution Pi (1D array)
    W = np.sum(adjacencyMatrix)
    Pi = np.sum(adjacencyMatrix, axis=1) / W
    Pi = Pi.flatten()

    # ---------- Compute aggregation time matrix Eta ----------
    A = np.zeros((N * N, N * N))
    B = np.ones((N * N, 1)) / 2

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

    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]

    Eta = X.reshape(N, N)

    # ---------- Compute constant terms eta_1, eta_2, eta_3 ----------
    eta_1 = 0.0
    eta_2 = 0.0
    eta_3 = 0.0
    for i in range(N):
        for j in range(N):
            eta_1 += Pi[i] * P[i, j] * Eta[i, j]
            eta_2 += Pi[i] * P2[i, j] * Eta[i, j]
            eta_3 += Pi[i] * P3[i, j] * Eta[i, j]

    # ---------- Compute required coefficients ----------
    coeff_term2 = 0.0  # Corresponds to term2: ∑_{i,k} π_i P[i,j] P[j,k] η_{j,k}
    coeff_term4 = 0.0  # Corresponds to term4: ∑_{i,j,k} π_i P[i,j] P[i,l] P[l,k] η_{j,k}
    coeff_term6 = 0.0  # Corresponds to term6: ∑_{i,k} π_i P[i,j] P2[j,k] η_{j,k}
    coeff_term8 = 0.0  # Corresponds to term8: ∑_{i,j,k} π_i P[i,j] P[i,l] P2[l,k] η_{j,k}

    j_node = lambda_node
    l_node = lambda_node

    # term2
    for i in range(N):
        for k in range(N):
            coeff_term2 += Pi[i] * P[i, j_node] * P[j_node, k] * Eta[j_node, k]

    # term4
    for i in range(N):
        for j in range(N):
            for k in range(N):
                coeff_term4 += Pi[i] * P[i, j] * P[i, l_node] * P[l_node, k] * Eta[j, k]

    # term6
    for i in range(N):
        for k in range(N):
            coeff_term6 += Pi[i] * P[i, j_node] * P2[j_node, k] * Eta[j_node, k]

    # term8
    for i in range(N):
        for j in range(N):
            for k in range(N):
                coeff_term8 += Pi[i] * P[i, j] * P[i, l_node] * P2[l_node, k] * Eta[j, k]

    constants = {
        'eta_1': eta_1,
        'eta_2': eta_2,
        'eta_3': eta_3,
        'coeff_term2': coeff_term2,
        'coeff_term4': coeff_term4,
        'coeff_term6': coeff_term6,
        'coeff_term8': coeff_term8,
    }
    return constants


def compute_threshold_for_lambda(lambda_val, constants):
    """
    Quickly compute the threshold using precomputed constants and a given λ.
    Numerator = η_2 - λ * coeff_term2 + λ * coeff_term4
    Denominator = (η_3 - η_1) - λ * coeff_term6 + λ * coeff_term8
    """
    numer = (constants['eta_2'] -
             constants['coeff_term2'] * lambda_val +
             constants['coeff_term4'] * lambda_val)

    denom = (constants['eta_3'] - constants['eta_1'] -
             constants['coeff_term6'] * lambda_val +
             constants['coeff_term8'] * lambda_val)

    if abs(denom) < 1e-12:
        return np.inf
    else:
        return numer / denom


def find_asymptotes(lambda_vals, thresholds, tolerance=0.05):
    """
    Find locations of asymptotes.
    """
    thresholds_array = np.array(thresholds)
    lambda_array = np.array(lambda_vals)

    finite_mask = np.isfinite(thresholds_array)
    if not np.any(finite_mask):
        return None, None

    finite_thresholds = thresholds_array[finite_mask]
    finite_lambdas = lambda_array[finite_mask]

    vertical_asymptote = None
    if len(finite_thresholds) > 1:
        threshold_diff = np.abs(np.diff(finite_thresholds))
        lambda_diff = np.abs(np.diff(finite_lambdas))

        if len(threshold_diff) > 0:
            change_rate = threshold_diff / (lambda_diff + 1e-12)
            max_change_idx = np.argmax(change_rate)

            if change_rate[max_change_idx] > 1 / tolerance:
                vertical_asymptote = (finite_lambdas[max_change_idx] +
                                      finite_lambdas[max_change_idx + 1]) / 2

    horizontal_asymptote = None
    pos_mask = finite_lambdas > 0
    if np.any(pos_mask):
        pos_lambdas = finite_lambdas[pos_mask]
        pos_thresholds = finite_thresholds[pos_mask]
        if len(pos_thresholds) >= 3:
            horizontal_asymptote = np.mean(pos_thresholds[-3:])

    if horizontal_asymptote is None:
        neg_mask = finite_lambdas < 0
        if np.any(neg_mask):
            neg_lambdas = finite_lambdas[neg_mask]
            neg_thresholds = finite_thresholds[neg_mask]
            if len(neg_thresholds) >= 3:
                horizontal_asymptote = np.mean(neg_thresholds[:3])

    return vertical_asymptote, horizontal_asymptote


def run_calculations_for_network_type(graph_type, n_values, lambda_range=(-10, 10), lambda_step=0.5):
    """
    Run calculations for a given network type and multiple network sizes (optimized version).
    """
    results_dict = {}

    for n in n_values:
        print(f"\nStarting calculation for network size n = {n}")

        # Set random seed for reproducibility
        final_seed = 42

        # Generate network
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
        else:
            raise ValueError(f"Unknown network type: {graph_type}")

        print(f"Network type: {graph_type}, nodes: {n}, edges: {G.number_of_edges()}")

        # Fix the node with index 1 as the one that carries λ
        lambda_node = 1
        if lambda_node >= n:
            lambda_node = 0
        print(f"Using node {lambda_node} as the node carrying lambda value")

        # Precompute network constants (λ-independent part)
        constants = precompute_network_constants(G, lambda_node)

        # Generate λ values
        lambda_start, lambda_end = lambda_range
        lambda_values = np.arange(lambda_start, lambda_end + lambda_step / 2, lambda_step)
        thresholds = []

        # Iterate over all λ values, compute thresholds quickly
        for idx, lambda_val in enumerate(lambda_values):
            try:
                threshold = compute_threshold_for_lambda(lambda_val, constants)
            except Exception as e:
                print(f"Error when computing λ={lambda_val}: {e}")
                threshold = np.inf

            thresholds.append(threshold)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(lambda_values) - 1:
                status = "∞" if threshold == np.inf else f"{threshold:.6f}"
                print(f"Progress: {idx+1:3d}/{len(lambda_values):3d} | λ = {lambda_val:5.1f} | threshold = {status}")

        # Find asymptotes
        vertical_asymptote, horizontal_asymptote = find_asymptotes(lambda_values, thresholds)

        results_dict[n] = {
            'lambda_vals': list(lambda_values),
            'hom_thresholds': list(thresholds),
            'vertical_asymptote': vertical_asymptote,
            'horizontal_asymptote': horizontal_asymptote
        }

        print(f"Finished calculation for n={n}")
        if vertical_asymptote is not None:
            print(f"Vertical asymptote position: λ = {vertical_asymptote:.4f}")
        if horizontal_asymptote is not None:
            print(f"Horizontal asymptote position: b/c = {horizontal_asymptote:.4f}")

    return results_dict, filename_prefix


def save_results_pickle(results_dict, filename_prefix, network_type):
    """
    Save results as pickle file.
    """
    filename = f"{network_type}_homogeneous_results_new.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"Results saved to: {filename}")

    for n, result in results_dict.items():
        csv_filename = f"direct_calculation_{filename_prefix}_{n}.csv"
        with open(csv_filename, 'w') as f:
            f.write("lambda,threshold\n")
            for lambda_val, threshold in zip(result['lambda_vals'], result['hom_thresholds']):
                if threshold == np.inf or threshold == float('inf'):
                    threshold_str = "inf"
                else:
                    threshold_str = f"{threshold:.6f}"
                f.write(f"{lambda_val:.6f},{threshold_str}\n")
        print(f"CSV results saved to: {csv_filename}")


def main():
    """
    Main function: compute results for multiple network sizes.
    """
    # Parameter settings
    network_type = "ba"  # options: "ws", "ba", "rg"
    n_values = [20, 50, 100, 200]  # list of network sizes

    # Set lambda range and step size
    lambda_range = (-10, 10)
    lambda_step = 0.5

    print("=" * 60)
    print("Starting calculation of aggregation time threshold")
    print(f"Network type: {network_type}")
    print(f"Network sizes: {n_values}")
    print(f"lambda range: [{lambda_range[0]}, {lambda_range[1]}]")
    print(f"lambda step: {lambda_step}")
    print("=" * 60)

    # Run calculations
    results_dict, filename_prefix = run_calculations_for_network_type(
        network_type, n_values, lambda_range, lambda_step
    )

    # Save results
    save_results_pickle(results_dict, filename_prefix, network_type)

    # Print result summary
    print("\n" + "=" * 60)
    print("Result summary:")
    for n, result in results_dict.items():
        finite_thresholds = [t for t in result['hom_thresholds'] if t != np.inf and t != float('inf')]
        if finite_thresholds:
            min_threshold = min(finite_thresholds)
            max_threshold = max(finite_thresholds)
            print(f"n={n}: threshold range [{min_threshold:.4f}, {max_threshold:.4f}]")
            if result['vertical_asymptote']:
                print(f"     Vertical asymptote: λ = {result['vertical_asymptote']:.4f}")
            if result['horizontal_asymptote']:
                print(f"     Horizontal asymptote: b/c = {result['horizontal_asymptote']:.4f}")
        else:
            print(f"n={n}: All thresholds are infinite")

    print("\nCalculation complete! You can use the plotting code to visualize the results.")
    print(f"Plotting code will read file: {network_type}_homogeneous_results_new.pkl")


if __name__ == "__main__":
    main()
