import networkx as nx
import numpy as np
import warnings
import pickle
from scipy import stats

warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")

def generate_lambda_distribution(N, distribution_type, params=None, seed=42):
    """生成 lambda 分布（可复现）"""
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
        raise ValueError(f"不支持的分布类型: {distribution_type}")
    
    if np.linalg.norm(lambda_array) > 0:
        lambda_array = lambda_array / np.linalg.norm(lambda_array) * np.sqrt(N)
    return lambda_array

def compute_coefficients_direct(G, base_lambda_array):
    """
    计算常数项 eta_1, eta_2, eta_3 以及与 lambda 有关的简化项：
        coeff2, coeff4, coeff6, coeff8
    其他项在阈值公式中相互抵消，不再计算。
    """
    adj = nx.adjacency_matrix(G).todense()
    N = len(G.nodes)
    
    # 转移概率矩阵 P
    P = np.zeros((N, N))
    for i in range(N):
        deg = np.sum(adj[i])
        if deg > 0:
            P[i] = adj[i] / deg
    
    P2 = P @ P
    P3 = P @ P @ P
    
    # 平稳分布 Pi
    Pi = np.zeros(N)
    W = np.sum(adj)
    for i in range(N):
        Pi[i] = np.sum(adj[i]) / W
    
    # 构建线性系统求 Eta
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
            # 基于 j 的转移
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    A[row, k] = 1 - 0.5 * P[j, k % N]
                else:
                    A[row, k] = -0.5 * P[j, k % N]
            # 基于 i 的转移
            for k in range(j * N, (j + 1) * N):
                A[row, k] = -0.5 * P[i, k % N]
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
    Eta = X.reshape(N, N)
    
    # 常数项
    eta_1 = 0.0
    eta_2 = 0.0
    eta_3 = 0.0
    for i in range(N):
        for j in range(N):
            eta_1 += Pi[i] * P[i, j] * Eta[i, j]
            eta_2 += Pi[i] * P2[i, j] * Eta[i, j]
            eta_3 += Pi[i] * P3[i, j] * Eta[i, j]
    
    # 只计算最终需要的与 lambda 有关的项
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
    """使用简化系数计算阈值"""
    eta_1 = coeffs['eta_1']
    eta_2 = coeffs['eta_2']
    eta_3 = coeffs['eta_3']
    # 原公式中 c1,c3,c5,c7 均被抵消，仅保留以下四项
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
    """查找渐近线（与原代码相同）"""
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
    """主计算流程（可复现）"""
    results_dict = {}
    for n in n_values:
        print(f"\n开始计算: 分布={distribution_type}, 网络={graph_type}, 规模={n}")
        # 生成连通图
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
            raise ValueError(f"未知的网络类型: {graph_type}")
        print(f"网络: {graph_type}, 节点数: {n}, 边数: {G.number_of_edges()}")
        
        # 生成基础 lambda 分布（固定种子）
        base_lambda = generate_lambda_distribution(n, distribution_type, seed=seed)
        lambda_stats = {
            'mean': np.mean(base_lambda), 'std': np.std(base_lambda),
            'min': np.min(base_lambda), 'max': np.max(base_lambda),
            'skewness': stats.skew(base_lambda), 'kurtosis': stats.kurtosis(base_lambda)
        }
        print(f"lambda统计: 均值={lambda_stats['mean']:.3f}, 标准差={lambda_stats['std']:.3f}")
        print(f"           最小值={lambda_stats['min']:.3f}, 最大值={lambda_stats['max']:.3f}")
        
        # 计算系数（仅包含简化后的四项）
        coeffs = compute_coefficients_direct(G, base_lambda)
        
        # 遍历缩放因子
        lambda_scales = np.arange(lambda_range[0], lambda_range[1] + lambda_step/2, lambda_step)
        thresholds = []
        for idx, scale in enumerate(lambda_scales):
            try:
                thr = compute_threshold_from_coefficients(coeffs, scale)
            except Exception as e:
                print(f"计算scale={scale}时出错: {e}")
                thr = np.inf
            thresholds.append(thr)
            if (idx+1) % 10 == 0 or idx == 0 or idx == len(lambda_scales)-1:
                status = "∞" if thr == np.inf else f"{thr:.6f}"
                print(f"进度: {idx+1:3d}/{len(lambda_scales):3d} | scale = {scale:5.1f} | 阈值 = {status}")
        
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
        print(f"完成 n={n} 的计算")
        if vert_asym: print(f"垂直渐近线位置: scale = {vert_asym:.4f}")
        if horz_asym: print(f"水平渐近线位置: b/c = {horz_asym:.4f}")
    return results_dict, cur_seed

def save_results_pickle(results_dict, distribution_type, graph_type):
    """保存结果到文件"""
    filename = f"{distribution_type}_{graph_type}_results.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"\n结果已保存到: {filename}")
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
        print(f"  Lambda值保存到: {lambda_fn}\n  阈值数据保存到: {csv_fn}")
    return filename

def main():
    distribution_types = ['uniform', 'normal', 'exponential', 'powerlaw']
    graph_types = ['ws']        # 可修改为 ['rg','ws','ba']
    n_values = [50]             # 网络规模
    lambda_range = (-2, 2)
    lambda_step = 0.5
    
    all_results = {}
    for dist in distribution_types:
        print("="*70)
        print(f"开始计算分布类型: {dist}")
        print("="*70)
        for gtype in graph_types:
            print(f"\n网络类型: {gtype}")
            print("-"*50)
            res_dict, _ = run_calculations_for_distribution(dist, gtype, n_values, lambda_range, lambda_step, seed=42)
            save_results_pickle(res_dict, dist, gtype)
            all_results[f"{dist}_{gtype}"] = res_dict
            print("\n结果摘要:")
            for n, r in res_dict.items():
                finite = [t for t in r['thresholds'] if t not in (np.inf, float('inf'))]
                if finite:
                    print(f"  n={n}: 阈值范围 [{min(finite):.4f}, {max(finite):.4f}]")
                    if r['vertical_asymptote']: print(f"        垂直渐近线: scale = {r['vertical_asymptote']:.4f}")
                    if r['horizontal_asymptote']: print(f"        水平渐近线: b/c = {r['horizontal_asymptote']:.4f}")
                else:
                    print(f"  n={n}: 所有阈值均为无穷大")
    
    with open("all_distributions_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    print("\n"+"="*70+"\n所有计算完成! 合并结果保存到: all_distributions_results.pkl")
    
    print("\nLambda分布统计摘要:")
    print("-"*70)
    for key, res_dict in all_results.items():
        print(f"\n分布: {key}")
        for n, r in res_dict.items():
            s = r['lambda_stats']
            print(f"  n={n}: 均值={s['mean']:.3f}, 标准差={s['std']:.3f}, 范围=[{s['min']:.3f},{s['max']:.3f}], 偏度={s['skewness']:.3f}, 峰度={s['kurtosis']:.3f}")

if __name__ == "__main__":
    main()