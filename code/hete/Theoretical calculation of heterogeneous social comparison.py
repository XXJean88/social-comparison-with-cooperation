import networkx as nx
import numpy as np
import warnings
import pickle
from scipy import stats

warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")

def generate_lambda_distribution(N, distribution_type, params=None, seed=42):
    """Generate lambda distribution (reproducible)"""
    np.random.seed(seed)
    default_params = {
        'uniform': {'low': -1, 'high': 1},
        'normal': {'mean': 0, 'std': 1},
        'exponential': {'scale': 1},
        'powerlaw': {'alpha': 2.5}
    }
    if params is None:
        params = default_params.get(distribution_type, {})
    
    if distribution_type == 'uniform':
        low = params.get('low', -1)
        high = params.get('high', 1)
        lambda_array = np.random.uniform(low, high, N)
    elif distribution_type == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        lambda_array = np.random.normal(mean, std, N)
    elif distribution_type == 'exponential':
        scale = params.get('scale', 1)
        exp_values = np.random.exponential(scale, N)
        signs = np.random.choice([-1, 1], N)
        lambda_array = exp_values * signs
    elif distribution_type == 'powerlaw':
        alpha = params.get('alpha', 2.5)
        uniform = np.random.uniform(0, 1, N)
        lambda_array = (1 - uniform) ** (-1 / (alpha - 1))
        signs = np.random.choice([-1, 1], N)
        lambda_array = lambda_array * signs
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    if np.linalg.norm(lambda_array) > 0:
        lambda_array = lambda_array / np.linalg.norm(lambda_array) * np.sqrt(N)
    return lambda_array

def compute_coefficients_direct(G, base_lambda_array):
    """
    Compute constant terms eta_1, eta_2, eta_3 and the simplified lambda-related terms:
        coeff2, coeff4, coeff6, coeff8
    Other terms cancel out in the threshold formula and are not computed.
    """
    adj = nx.adjacency_matrix(G).todense()
    N = len(G.nodes)
    
    # Transition probability matrix P
    P = np.zeros((N, N))
    for i in range(N):
        deg = np.sum(adj[i])
        if deg > 0:
            P[i] = adj[i] / deg
    
    P2 = P @ P
    P3 = P @ P @ P
    
    # Stationary distribution Pi
    Pi = np.zeros(N)
    W = np.sum(adj)
    for i in range(N):
        Pi[i] = np.sum(adj[i]) / W
    
    # Build linear system to solve for Eta
    A = np.zeros((N * N, N * N))
    B = np.ones((N * N, 1)) / 2
    diag_idx = [i * N + i for i in range(N)]
    for row in range(N * N):
        if row in diag_idx:
            A[row, row] = 1
            B[row, 0] = 0
        else:
            i = row // N
            j = row % N
            # Transition based on j
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    A[row, k] = 1 - 0.5 * P[j, k % N]
                else:
                    A[row, k] = -0.5 * P[j, k % N]
            # Transition based on i
            for k in range(j * N, (j + 1) * N):
                A[row, k] = -0.5 * P[i, k % N]
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
    Eta = X.reshape(N, N)
    
    # Constant terms
    eta_1 = 0.0
    eta_2 = 0.0
    eta_3 = 0.0
    for i in range(N):
        for j in range(N):
            eta_1 += Pi[i] * P[i, j] * Eta[i, j]
            eta_2 += Pi[i] * P2[i, j] * Eta[i, j]
            eta_3 += Pi[i] * P3[i, j] * Eta[i, j]
    
    # Compute only the lambda-related terms needed in the final formula
    coeff2 = 0.0   # ∑ λ_j π_i p_{ij} p_{jk} η_{jk}
    coeff4 = 0.0   # ∑ λ_l π_i p_{ij} p_{il} p_{lk} η_{jk}
    coeff6 = 0.0   # ∑ λ_j π_i p_{ij} p^{(2)}_{jk} η_{jk}
    coeff8 = 0.0   # ∑ λ_l π_i p_{ij} p_{il} p^{(2)}_{lk} η_{jk}
    
    # coeff2
    for i in range(N):
        for j in range(N):
            w_ij = Pi[i] * P[i, j]
            lj = base_lambda_array[j]
            for k in range(N):
                factor = w_ij * P[j, k]
                coeff2 += lj * factor * Eta[j, k]
    
    # coeff4
    for i in range(N):
        for j in range(N):
            w_ij = Pi[i] * P[i, j]
            for l in range(N):
                ll = base_lambda_array[l]
                w_il = P[i, l]
                for k in range(N):
                    factor = w_ij * w_il * P[l, k]
                    coeff4 += ll * factor * Eta[j, k]
    
    # coeff6
    for i in range(N):
        for j in range(N):
            w_ij = Pi[i] * P[i, j]
            lj = base_lambda_array[j]
            for k in range(N):
                factor = w_ij * P2[j, k]
                coeff6 += lj * factor * Eta[j, k]
    
    # coeff8
    for i in range(N):
        for j in range(N):
            w_ij = Pi[i] * P[i, j]
            for l in range(N):
                ll = base_lambda_array[l]
                w_il = P[i, l]
                for k in range(N):
                    factor = w_ij * w_il * P2[l, k]
                    coeff8 += ll * factor * Eta[j, k]
    
    return {
        'eta_1': eta_1,
        'eta_2': eta_2,
        'eta_3': eta_3,
        'coeff2': coeff2,
        'coeff4': coeff4,
        'coeff6': coeff6,
        'coeff8': coeff8,
    }

def compute_threshold_from_coefficients(coeffs, lambda_val):
    """Compute threshold using the simplified coefficients"""
    eta_1 = coeffs['eta_1']
    eta_2 = coeffs['eta_2']
    eta_3 = coeffs['eta_3']
    # In the original formula, c1, c3, c5, c7 cancel out, only the following four terms remain
    c2 = coeffs['coeff2']
    c4 = coeffs['coeff4']
    c6 = coeffs['coeff6']
    c8 = coeffs['coeff8']
    
    numerator = eta_2 + lambda_val * (-c2 + c4)
    denominator = eta_3 - eta_1 + lambda_val * (-c6 + c8)
    
    if abs(denominator) < 1e-12:
        return np.inf
    return numerator / denominator

def find_asymptotes(lambda_vals, thresholds, tolerance=0.05):
    """Find asymptotes (same as original code)"""
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
            if change_rate[max_change_idx] > 1/tolerance:
                vertical_asymptote = (finite_lambdas[max_change_idx] + finite_lambdas[max_change_idx + 1]) / 2
    horizontal_asymptote = None
    pos_mask = finite_lambdas > 0
    if np.any(pos_mask):
        pos_thresholds = finite_thresholds[pos_mask]
        if len(pos_thresholds) >= 3:
            horizontal_asymptote = np.mean(pos_thresholds[-3:])
    if horizontal_asymptote is None:
        neg_mask = finite_lambdas < 0
        if np.any(neg_mask):
            neg_thresholds = finite_thresholds[neg_mask]
            if len(neg_thresholds) >= 3:
                horizontal_asymptote = np.mean(neg_thresholds[:3])
    return vertical_asymptote, horizontal_asymptote

def run_calculations_for_distribution(distribution_type, graph_type, n_values, lambda_range=(-10,10), lambda_step=0.5, seed=42):
    """Main computation workflow (reproducible)"""
    results_dict = {}
    for n in n_values:
        print(f"\nStarting: distribution={distribution_type}, network={graph_type}, size={n}")
        # Generate connected graph
        cur_seed = seed
        if graph_type == "rg":
            G = nx.random_graphs.random_regular_graph(4, n, seed=cur_seed)
            while not nx.is_connected(G):
                cur_seed += 1
                G = nx.random_graphs.random_regular_graph(4, n, seed=cur_seed)
        elif graph_type == "ba":
            G = nx.barabasi_albert_graph(n=n, m=2, seed=cur_seed)
            while not nx.is_connected(G):
                cur_seed += 1
                G = nx.barabasi_albert_graph(n=n, m=2, seed=cur_seed)
        elif graph_type == "ws":
            G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=cur_seed)
            while not nx.is_connected(G):
                cur_seed += 1
                G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=cur_seed)
        else:
            raise ValueError(f"Unknown network type: {graph_type}")
        print(f"Network: {graph_type}, nodes: {n}, edges: {G.number_of_edges()}")
        
        # Generate base lambda distribution (fixed seed)
        base_lambda = generate_lambda_distribution(n, distribution_type, seed=seed)
        lambda_stats = {
            'mean': np.mean(base_lambda), 'std': np.std(base_lambda),
            'min': np.min(base_lambda), 'max': np.max(base_lambda),
            'skewness': stats.skew(base_lambda), 'kurtosis': stats.kurtosis(base_lambda)
        }
        print(f"Lambda stats: mean={lambda_stats['mean']:.3f}, std={lambda_stats['std']:.3f}")
        print(f"            min={lambda_stats['min']:.3f}, max={lambda_stats['max']:.3f}")
        
        # Compute coefficients (only the simplified four terms)
        coeffs = compute_coefficients_direct(G, base_lambda)
        
        # Iterate over scaling factors
        lambda_scales = np.arange(lambda_range[0], lambda_range[1] + lambda_step/2, lambda_step)
        thresholds = []
        for idx, scale in enumerate(lambda_scales):
            try:
                thr = compute_threshold_from_coefficients(coeffs, scale)
            except Exception as e:
                print(f"Error computing scale={scale}: {e}")
                thr = np.inf
            thresholds.append(thr)
            if (idx+1) % 10 == 0 or idx == 0 or idx == len(lambda_scales)-1:
                status = "inf" if thr == np.inf else f"{thr:.6f}"
                print(f"Progress: {idx+1:3d}/{len(lambda_scales):3d} | scale = {scale:5.1f} | threshold = {status}")
        
        vert_asym, horz_asym = find_asymptotes(lambda_scales, thresholds)
        results_dict[n] = {
            'distribution_type': distribution_type,
            'graph_type': graph_type,
            'network_size': n,
            'lambda_scales': list(lambda_scales),
            'thresholds': list(thresholds),
            'base_lambda_values': list(base_lambda),
            'lambda_stats': lambda_stats,
            'vertical_asymptote': vert_asym,
            'horizontal_asymptote': horz_asym
        }
        print(f"Finished n={n}")
        if vert_asym: print(f"Vertical asymptote at scale = {vert_asym:.4f}")
        if horz_asym: print(f"Horizontal asymptote at b/c = {horz_asym:.4f}")
    return results_dict, cur_seed

def save_results_pickle(results_dict, distribution_type, graph_type):
    """Save results to files"""
    filename = f"{distribution_type}_{graph_type}_results.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"\nResults saved to: {filename}")
    for n, res in results_dict.items():
        lambda_fn = f"lambda_{distribution_type}_{graph_type}_n{n}.txt"
        with open(lambda_fn, 'w') as f:
            f.write(f"# Distribution: {distribution_type}\n# Graph type: {graph_type}\n# Network size: {n}\n# Lambda values:\n")
            for idx, val in enumerate(res['base_lambda_values']):
                f.write(f"{idx}\t{val:.6f}\n")
        csv_fn = f"thresholds_{distribution_type}_{graph_type}_n{n}.csv"
        with open(csv_fn, 'w') as f:
            f.write("scale,threshold\n")
            for scale, thr in zip(res['lambda_scales'], res['thresholds']):
                thr_str = "inf" if thr in (np.inf, float('inf')) else f"{thr:.6f}"
                f.write(f"{scale:.6f},{thr_str}\n")
        print(f"  Lambda values saved to: {lambda_fn}\n  Threshold data saved to: {csv_fn}")
    return filename

def main():
    distribution_types = ['uniform', 'normal', 'exponential', 'powerlaw']
    graph_types = ['ws']        # Can be changed to ['rg','ws','ba']
    n_values = [50]             # Network sizes
    lambda_range = (-2, 2)
    lambda_step = 0.5
    
    all_results = {}
    for dist in distribution_types:
        print("="*70)
        print(f"Starting distribution: {dist}")
        print("="*70)
        for gtype in graph_types:
            print(f"\nNetwork type: {gtype}")
            print("-"*50)
            res_dict, _ = run_calculations_for_distribution(dist, gtype, n_values, lambda_range, lambda_step, seed=42)
            save_results_pickle(res_dict, dist, gtype)
            all_results[f"{dist}_{gtype}"] = res_dict
            print("\nSummary:")
            for n, r in res_dict.items():
                finite = [t for t in r['thresholds'] if t not in (np.inf, float('inf'))]
                if finite:
                    print(f"  n={n}: threshold range [{min(finite):.4f}, {max(finite):.4f}]")
                    if r['vertical_asymptote']: print(f"        vertical asymptote: scale = {r['vertical_asymptote']:.4f}")
                    if r['horizontal_asymptote']: print(f"        horizontal asymptote: b/c = {r['horizontal_asymptote']:.4f}")
                else:
                    print(f"  n={n}: all thresholds are infinite")
    
    with open("all_distributions_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    print("\n"+"="*70+"\nAll calculations completed! Combined results saved to: all_distributions_results.pkl")
    
    print("\nLambda distribution statistics summary:")
    print("-"*70)
    for key, res_dict in all_results.items():
        print(f"\nDistribution: {key}")
        for n, r in res_dict.items():
            s = r['lambda_stats']
            print(f"  n={n}: mean={s['mean']:.3f}, std={s['std']:.3f}, range=[{s['min']:.3f},{s['max']:.3f}], skewness={s['skewness']:.3f}, kurtosis={s['kurtosis']:.3f}")

if __name__ == "__main__":
    main()
