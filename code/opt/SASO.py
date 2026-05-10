import numpy as np
import networkx as nx
from scipy.linalg import solve
import multiprocessing as mp
import pickle
import os
import json

# ==============================
# 1. 构建固定网络（改为N=50）
# ==============================

N = 50   # 与可视化代码中的 n=50 保持一致
G = nx.random_graphs.random_regular_graph(4, N, seed=42)
# G = nx.barabasi_albert_graph(N, m=2, seed=42)
# G = nx.watts_strogatz_graph(N, k=4, p=0.3, seed=42)

A = nx.to_numpy_array(G)
P = A / A.sum(axis=1, keepdims=True)
P2 = P @ P
P3 = P2 @ P

deg = A.sum(axis=1)
Pi = deg / deg.sum()

# ==============================
# 2. 计算聚合时间 Eta (只做一次)
# ==============================

def compute_eta(P):
    N = P.shape[0]
    size = N*N
    A_mat = np.zeros((size, size))
    b = np.ones(size) / 2

    for i in range(N):
        for j in range(N):
            idx = i*N + j
            if i == j:
                A_mat[idx, idx] = 1
                b[idx] = 0
            else:
                for k in range(N):
                    A_mat[idx, i*N+k] -= 0.5 * P[j, k]
                    A_mat[idx, j*N+k] -= 0.5 * P[i, k]
                A_mat[idx, idx] += 1

    eta = solve(A_mat, b)
    return eta.reshape(N, N)

print("Computing coalescence times...")
Eta = compute_eta(P)
print("Done.")

# ==============================
# 3. 预计算 A0, B0, C, D
# ==============================

eta1 = np.sum(Pi[:,None] * P * Eta)
eta2 = np.sum(Pi[:,None] * P2 * Eta)
eta3 = np.sum(Pi[:,None] * P3 * Eta)

A0 = eta2
B0 = eta3 - eta1

C = np.zeros(N)
D = np.zeros(N)

for j in range(N):
    C[j] = np.sum(Pi[:,None] * P[:,j][:,None] * P[j,:][None,:] *
                  (Eta - Eta[j,:][None,:]))
    D[j] = np.sum(Pi[:,None] * P[:,j][:,None] * P2[j,:][None,:] *
                  (Eta - Eta[j,:][None,:]))

print("Precomputation finished.")

# ==============================
# 4. 阈值函数（O(N)）
# ==============================

def threshold_batch(L):
    num = A0 + L @ C
    den = B0 + L @ D
    out = np.full(len(L), np.inf)
    mask = den != 0
    val = num[mask] / den[mask]
    val[val <= 0] = np.inf
    out[mask] = val
    return out

# 初始阈值（λ = 0）
init_lambda = np.zeros(N)
init_threshold = threshold_batch(init_lambda.reshape(1,-1))[0]
print("Initial threshold:", init_threshold)

# ==============================
# 5. PSO 优化（带历史记录）
# ==============================

def PSO_optimize(max_iter=150, swarm_size=40):
    # 历史记录容器
    history = {
        'thresholds': [],               # 每代全局最优阈值
        'best_lambda_history': [],      # 每代全局最优λ向量
        'lambda_mean_history': [],      # 每代所有粒子λ的均值（N维）
        'lambda_std_history': []        # 每代所有粒子λ的标准差（N维）
    }

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    pos = np.zeros((swarm_size, N))                     # 位置全零
    vel = np.random.randn(swarm_size, N) * 0.1          # 速度小随机数

    pbest = pos.copy()
    pbest_val = threshold_batch(pos)

    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    print("\nStart PSO optimization")
    print(f"Iter 0: threshold = {gbest_val:.6f}")

    # 记录第0代
    history['thresholds'].append(gbest_val)
    history['best_lambda_history'].append(gbest.copy())
    history['lambda_mean_history'].append(np.mean(pos, axis=0))
    history['lambda_std_history'].append(np.std(pos, axis=0))

    for it in range(1, max_iter+1):
        r1 = np.random.rand(swarm_size, N)
        r2 = np.random.rand(swarm_size, N)

        vel = (w*vel
               + c1*r1*(pbest - pos)
               + c2*r2*(gbest - pos))

        pos += vel

        values = threshold_batch(pos)

        better = values < pbest_val
        pbest[better] = pos[better]
        pbest_val[better] = values[better]

        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        # 记录当前代
        history['thresholds'].append(gbest_val)
        history['best_lambda_history'].append(gbest.copy())
        history['lambda_mean_history'].append(np.mean(pos, axis=0))
        history['lambda_std_history'].append(np.std(pos, axis=0))

        if it % 10 == 0 or it == 1:
            print(f"Iter {it}: threshold = {gbest_val:.6f}")
    return gbest, gbest_val, history

# 运行优化
best_lambda, best_threshold, history = PSO_optimize()

print("\nOptimization finished.")
print("Final threshold:", best_threshold)
print("Initial threshold (lambda=0):", init_threshold)

# ==============================
# 6. 保存优化过程数据
# ==============================

def save_pso_data(N, G, history, best_lambda, best_threshold, init_threshold):
    network_type = 'RG'

    # 创建目录
    save_dir = f"{network_type}_{N}_network_pso_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 打包数据
    data = {
        'network_info': {
            'type': network_type,
            'nodes': N,
            'avg_degree': np.mean(list(dict(G.degree()).values())),
            'seed': 42
        },
        'optimization_params': {
            'max_iter': 150,
            'swarm_size': 40,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        },
        'particle_swarm': {
            'best_lambda': best_lambda.tolist(),   # 转为列表便于JSON序列化
            'best_threshold': float(best_threshold),
            'init_threshold': float(init_threshold),
            'history': {
                'thresholds': [float(t) if np.isfinite(t) else None for t in history['thresholds']],
                'best_lambda_history': [l.tolist() for l in history['best_lambda_history']],
                'lambda_mean_history': [m.tolist() for m in history['lambda_mean_history']],
                'lambda_std_history': [s.tolist() for s in history['lambda_std_history']]
            }
        }
    }

    # 保存为 pickle（完整数据）
    pkl_path = os.path.join(save_dir, f"{network_type}_{N}_full_data.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Full data saved to {pkl_path}")

    # 同时保存一份 JSON（便于快速查看，但会丢失 numpy 类型）
    json_path = os.path.join(save_dir, f"{network_type}_{N}_particle_swarm_results.json")
    with open(json_path, 'w') as f:
        json.dump(data['particle_swarm'], f, indent=2)
    print(f"JSON summary saved to {json_path}")

# 执行保存
save_pso_data(N, G, history, best_lambda, best_threshold, init_threshold)

print("\nAll data saved. You can now run the visualization code.")