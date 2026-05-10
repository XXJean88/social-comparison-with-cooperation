import networkx as nx
import numpy as np
import warnings
import pickle

# 忽略NumPy兼容性警告
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")


def precompute_network_constants(G, lambda_node):
    """
    预计算与 λ 无关的网络常数，以及所需的线性系数。
    """
    # 将邻接矩阵转换为 ndarray，避免 matrix 的奇怪行为
    adjacencyMatrix = np.asarray(nx.adjacency_matrix(G).todense())
    N = len(G.nodes)

    # 计算转移概率矩阵 P
    P = np.zeros((N, N))
    for row in range(N):
        sumOverRow = np.sum(adjacencyMatrix[row])
        if sumOverRow > 0:
            P[row, :] = adjacencyMatrix[row] / sumOverRow

    # 高阶转移概率
    P2 = P @ P
    P3 = P2 @ P

    # 平稳分布 Pi (一维数组)
    W = np.sum(adjacencyMatrix)
    Pi = np.sum(adjacencyMatrix, axis=1) / W
    Pi = Pi.flatten()

    # ---------- 计算聚合时间矩阵 Eta ----------
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
            # 第一个块：基于j的转移
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    A[row, k] = 1 - 0.5 * P[j, k % N]
                else:
                    A[row, k] = -0.5 * P[j, k % N]
            # 第二个块：基于i的转移
            for k in range(j * N, (j + 1) * N):
                A[row, k] = -0.5 * P[i, k % N]

    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]

    Eta = X.reshape(N, N)

    # ---------- 计算常数项 eta_1, eta_2, eta_3 ----------
    eta_1 = 0.0
    eta_2 = 0.0
    eta_3 = 0.0
    for i in range(N):
        for j in range(N):
            eta_1 += Pi[i] * P[i, j] * Eta[i, j]
            eta_2 += Pi[i] * P2[i, j] * Eta[i, j]
            eta_3 += Pi[i] * P3[i, j] * Eta[i, j]

    # ---------- 计算所需的系数 ----------
    coeff_term2 = 0.0  # 对应 term2: ∑_{i,k} π_i P[i,j] P[j,k] η_{j,k}
    coeff_term4 = 0.0  # 对应 term4: ∑_{i,j,k} π_i P[i,j] P[i,l] P[l,k] η_{j,k}
    coeff_term6 = 0.0  # 对应 term6: ∑_{i,k} π_i P[i,j] P2[j,k] η_{j,k}
    coeff_term8 = 0.0  # 对应 term8: ∑_{i,j,k} π_i P[i,j] P[i,l] P2[l,k] η_{j,k}

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
    使用预计算的常数和 λ 值快速计算阈值。
    分子 = η_2 - λ * coeff_term2 + λ * coeff_term4
    分母 = (η_3 - η_1) - λ * coeff_term6 + λ * coeff_term8
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
    查找渐近线位置
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
    为指定网络类型和多个网络规模运行计算（优化版本）。
    """
    results_dict = {}

    for n in n_values:
        print(f"\n开始计算网络规模 n = {n}")

        # 设置随机种子以确保可重复性
        final_seed = 42

        # 生成网络
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
            raise ValueError(f"未知的网络类型: {graph_type}")

        print(f"网络类型: {graph_type}, 节点数: {n}, 边数: {G.number_of_edges()}")

        # 固定使用索引为1的节点作为有lambda值的节点
        lambda_node = 1
        if lambda_node >= n:
            lambda_node = 0
        print(f"使用节点 {lambda_node} 作为有lambda值的节点")

        # 预计算网络常数（与 λ 无关的部分）
        constants = precompute_network_constants(G, lambda_node)

        # 生成λ值范围
        lambda_start, lambda_end = lambda_range
        lambda_values = np.arange(lambda_start, lambda_end + lambda_step / 2, lambda_step)
        thresholds = []

        # 遍历所有λ值，快速计算阈值
        for idx, lambda_val in enumerate(lambda_values):
            try:
                threshold = compute_threshold_for_lambda(lambda_val, constants)
            except Exception as e:
                print(f"计算λ={lambda_val}时出错: {e}")
                threshold = np.inf

            thresholds.append(threshold)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(lambda_values) - 1:
                status = "∞" if threshold == np.inf else f"{threshold:.6f}"
                print(f"进度: {idx+1:3d}/{len(lambda_values):3d} | λ = {lambda_val:5.1f} | 阈值 = {status}")

        # 查找渐近线
        vertical_asymptote, horizontal_asymptote = find_asymptotes(lambda_values, thresholds)

        results_dict[n] = {
            'lambda_vals': list(lambda_values),
            'hom_thresholds': list(thresholds),
            'vertical_asymptote': vertical_asymptote,
            'horizontal_asymptote': horizontal_asymptote
        }

        print(f"完成 n={n} 的计算")
        if vertical_asymptote is not None:
            print(f"垂直渐近线位置: λ = {vertical_asymptote:.4f}")
        if horizontal_asymptote is not None:
            print(f"水平渐近线位置: b/c = {horizontal_asymptote:.4f}")

    return results_dict, filename_prefix


def save_results_pickle(results_dict, filename_prefix, network_type):
    """
    保存结果为pickle文件（与原代码相同，未修改）。
    """
    filename = f"{network_type}_homogeneous_results_new.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"结果已保存到: {filename}")

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
        print(f"CSV结果已保存到: {csv_filename}")


def main():
    """
    主函数：计算多个网络规模的结果
    """
    # 参数设置
    network_type = "ba"  # 可以选择 "ws", "ba", "rg"
    n_values = [20, 50, 100, 200]  # 网络规模列表

    # 设置lambda范围和步长
    lambda_range = (-10, 10)
    lambda_step = 0.5

    print("=" * 60)
    print("开始计算聚合时间阈值")
    print(f"网络类型: {network_type}")
    print(f"网络规模: {n_values}")
    print(f"lambda范围: [{lambda_range[0]}, {lambda_range[1]}]")
    print(f"lambda步长: {lambda_step}")
    print("=" * 60)

    # 运行计算
    results_dict, filename_prefix = run_calculations_for_network_type(
        network_type, n_values, lambda_range, lambda_step
    )

    # 保存结果
    save_results_pickle(results_dict, filename_prefix, network_type)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("结果摘要:")
    for n, result in results_dict.items():
        finite_thresholds = [t for t in result['hom_thresholds'] if t != np.inf and t != float('inf')]
        if finite_thresholds:
            min_threshold = min(finite_thresholds)
            max_threshold = max(finite_thresholds)
            print(f"n={n}: 阈值范围 [{min_threshold:.4f}, {max_threshold:.4f}]")
            if result['vertical_asymptote']:
                print(f"     垂直渐近线: λ = {result['vertical_asymptote']:.4f}")
            if result['horizontal_asymptote']:
                print(f"     水平渐近线: b/c = {result['horizontal_asymptote']:.4f}")
        else:
            print(f"n={n}: 所有阈值均为无穷大")

    print("\n计算完成！您可以使用绘图代码绘制结果。")
    print(f"绘图代码将读取文件: {network_type}_homogeneous_results_new.pkl")


if __name__ == "__main__":
    main()