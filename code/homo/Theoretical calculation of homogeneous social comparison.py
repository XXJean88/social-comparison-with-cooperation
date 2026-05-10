import networkx as nx
import numpy as np
import warnings

# 忽略NumPy兼容性警告
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")

def Coalescence_Times_Homogeneous_Lambda(G, lambda_val):
    """
    同质性λ的聚合时间计算函数
    """
    adjacencyMatrix = nx.adjacency_matrix(G).todense()
    N = len(G.nodes)
    
    # 计算转移概率矩阵
    P = np.zeros((N, N))
    for row in range(N):
        sumOverRow = np.sum(adjacencyMatrix[row])
        for col in range(N):
            P[row, col] = adjacencyMatrix[row, col] / sumOverRow
    
    # 计算高阶转移概率
    P2 = P @ P
    P3 = P @ P @ P
    P4 = P @ P @ P @ P
    
    # 计算平稳分布
    Pi = np.zeros(N)
    W = np.sum(adjacencyMatrix)
    for i in range(N):
        Pi[i] = np.sum(adjacencyMatrix[i]) / W

    # 构建线性系统
    A = np.zeros((N * N, N * N))
    B = np.ones((N * N, 1)) / 2
    
    # 对角线元素处理
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

    # 求解线性系统
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
    
    Eta = X.reshape(N, N)

    # 计算聚合时间项
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
    
    # 使用同质性λ的公式计算阈值
    numerator = eta_2 + lambda_val * (eta_3 - eta_1)
    denominator = eta_3 - eta_1 + lambda_val * (eta_4 - eta_2)
    
    if abs(denominator) < 1e-12:
        return np.inf
    else:
        threshold = numerator / denominator
        return threshold

# 测试代码
n = 100
final_seed = 42

# 选择网络类型
graph_type = "ws"  # 可以改为 "rg", "ba", 或 "ws"

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
    
print(f"最终使用的随机种子: {final_seed}")
print(f"网络类型: {graph_type}")
print(f"网络节点数: {n}")
print(f"网络边数: {G.number_of_edges()}")

# 生成λ值范围
lambda_values = np.arange(-10, 10, 0.5)
thresholds = []

# 遍历所有λ值
for idx, lambda_val in enumerate(lambda_values):
    threshold = Coalescence_Times_Homogeneous_Lambda(G, lambda_val)
    thresholds.append(threshold)
    
    # 显示进度
    status = "∞" if threshold == np.inf else f"{threshold:.6f}"
    print(f"进度: {idx+1:2d}/{len(lambda_values):2d} | λ = {lambda_val:5.1f} | 阈值 = {status}")

# 输出结果摘要
finite_thresholds = [t for t in thresholds if t != np.inf]
finite_lambdas = [lambda_values[i] for i, t in enumerate(thresholds) if t != np.inf]

# 保存结果到文件
results = []
for lambda_val, threshold in zip(lambda_values, thresholds):
    results.append([lambda_val, threshold if threshold != np.inf else float('inf')])

results_array = np.array(results)
filename = f'homogeneous_lambda_{filename_prefix}_{n}.csv'
np.savetxt(filename, results_array, delimiter=',', 
            header='lambda,threshold', comments='', fmt='%.6f')
print(f"结果已保存到: {filename}")