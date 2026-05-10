import networkx as nx
import random
import numpy as np
from numba import jit
import multiprocessing
import functools
import math
import pandas as pd

@jit(nopython=True)  # 使用Numba加速
def single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array):
    """每个节点只随机选择一个邻居进行博弈"""
    for i in range(len(state_array)):  # 遍历所有节点
        # 随机选择一个邻居
        nbrs_num = deg_array[i]  # 节点i的邻居数量
        nbrs_array = nbr_mat[i][:nbrs_num]  # 获取邻居列表
        
        random_idx = random.randint(0, nbrs_num - 1)  # 随机选择邻居索引
        random_neighbor = nbrs_array[random_idx]  # 获取随机邻居
        
        # 只与随机选择的邻居进行博弈
        payoff_array[i] = game_matrix[state_array[i]][state_array[random_neighbor]]
    return payoff_array  # 返回收益数组

@jit(nopython=True)
def calculate_fitness(payoff_array, nbr_mat, deg_array, node_idx, w, lambda_param, use_neighbors):
    """
    计算适应度: 
    如果use_neighbors=True: F_i(x) = exp(w * (u_i(x) + λ * sum(p_im * u_m(x))))
    如果use_neighbors=False: F_i(x) = exp(w * u_i(x))
    """
    # 节点自身的收益
    self_payoff = payoff_array[node_idx]
    
    if use_neighbors:
        # 获取邻居
        nbrs_num = deg_array[node_idx]  # 邻居数量
        nbrs_array = nbr_mat[node_idx][:nbrs_num]  # 邻居列表
        
        # 计算邻居的平均收益
        neighbors_avg_payoff = np.sum(payoff_array[nbrs_array]) / nbrs_num
        
        # 计算适应度（包含邻居收益）
        fitness = math.exp(w * (self_payoff + lambda_param * neighbors_avg_payoff))
    else:
        # 计算适应度（只包含自身收益）
        fitness = math.exp(w * self_payoff)
    
    return fitness

@jit(nopython=True)
def strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node):
    """策略更新"""
    # 随机选择一个个体进行更新（所有个体更新速率相同）
    update_node = random.randint(0, nodesnum - 1)
    
    # 获取邻居信息
    nbrs_num = deg_array[update_node]  # 邻居数量
    nbrs_array = nbr_mat[update_node][:nbrs_num]  # 邻居列表
    
    # 计算所有邻居的适应度
    fitness_array = np.zeros(nbrs_num)  # 初始化适应度数组
    for i in range(nbrs_num):
        neighbor_idx = nbrs_array[i]  # 邻居索引
        
        # 判断该邻居是否是特殊节点（固定为索引1的节点）
        use_neighbors = (neighbor_idx == special_node)
        
        fitness_array[i] = calculate_fitness(payoff_array, nbr_mat, deg_array, neighbor_idx, w, lambda_param, use_neighbors)
    
    # 根据适应度选择策略
    total_fitness = np.sum(fitness_array)  # 总适应度
    if total_fitness > 0:
        prob_array = fitness_array / total_fitness  # 计算选择概率
        chosen_neighbor = rand_pick_list(nbrs_array, prob_array)  # 根据概率选择邻居
        state_array[update_node] = state_array[chosen_neighbor]  # 更新策略
    
    return state_array

@jit(nopython=True)
def evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node):
    total_generation = int(1e9)  # 总演化代数
    payoff_array = np.zeros(nodesnum, dtype=np.float64)  # 收益数组
    state_array = np.zeros(nodesnum, dtype=np.int_)  # 状态数组（0:背叛, 1:合作）
    coop_ini = np.random.choice(nodesnum)  # 随机选择一个初始合作者
    state_array[coop_ini] = 1  # 设置初始合作者
    
    for time in range(total_generation):
        # 进行单轮博弈和策略更新
        payoff_array = single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array)
        state_array = strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node)
        payoff_array[:] = 0  # 重置收益数组
        
        coord = np.sum(state_array)  # 计算合作者数量
        
        # 检查是否达到吸收状态
        if coord > nodesnum - 1:  # 所有节点都合作
            return 1
        if coord == 0:  # 所有节点都背叛
            return 0
            
    return coord / nodesnum  # 返回合作者比例

@jit(nopython=True)
def process(core, b, nbr_mat, deg_array, nodesnum, lambda_param, special_node):
    w = 0.01  # 选择强度参数
    # 定义博弈矩阵（囚徒困境）
    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0      # 双方背叛
    game_matrix[0][1] = b      # 自己背叛，对方合作
    game_matrix[1][0] = -1     # 自己合作，对方背叛  
    game_matrix[1][1] = b - 1  # 双方合作
    
    repeat_time = int(1e6)  # 重复次数
    repeat_array = np.zeros(repeat_time)  # 结果数组
    
    for rep in range(repeat_time):
        # 运行演化过程
        freq_c = evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_param, special_node)
        repeat_array[rep] = freq_c
    
    # 计算合作固定次数和总吸收次数
    coop_fixations = np.sum(repeat_array == 1)
    total_absorptions = np.sum(repeat_array == 1) + np.sum(repeat_array == 0)
    
    return coop_fixations, total_absorptions

@jit(nopython=True)
def rand_pick_list(pick_list, prob_list):
    """根据概率分布从列表中选择一个元素"""
    x = random.uniform(0, 1)  # 生成随机数
    cumulative_probability = 0.0  # 累积概率
    for item, item_probability in zip(pick_list, prob_list):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
    return item

def edge_list_array(edge_list):
    """将边列表转换为numpy数组"""
    edge_mat = np.zeros([len(edge_list), 2], int)  # 创建边矩阵
    for i in range(len(edge_list)):
        edge_mat[i, :] = np.array(edge_list[i])  # 填充边数据
    return edge_mat

def nbr_dict_mat(nbr_dict):
    """将邻居字典转换为numpy数组"""
    nodesnum = len(nbr_dict)  # 节点数量
    nbr_mat = np.zeros([nodesnum, nodesnum], int)  # 邻居矩阵
    deg_array = np.zeros(nodesnum, int)  # 度数组
    
    for i, nbrs in nbr_dict.items():
        deg_array[i] = len(nbrs)  # 记录节点度数
        if len(nbrs) > 0:
            nbr_mat[i][:len(nbrs)] = np.array(nbrs)  # 填充邻居信息
            
    return nbr_mat, deg_array

if __name__ == "__main__":
    # 生成n=100、度为4的随机正则图
    n = 50
    # G = nx.random_graphs.random_regular_graph(4, n, seed=42)
    # G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=42)
    G = nx.barabasi_albert_graph(n=n, m=2, seed=42)

            
    # 提取网络信息
    edge_list = list(G.edges())  # 边列表
    nbrs_dict = nx.to_dict_of_lists(G)  # 邻居字典
    nbr_mat, deg_array = nbr_dict_mat(nbrs_dict)  # 转换为矩阵形式
    nodesnum = G.number_of_nodes()  # 节点数量

    # 定义λ参数（衡量邻居收益影响程度的参数）
    lambda_param = 5
    special_node = 1

    # 计算不同b值下的固定概率
    b_array = [6.5]  # 收益参数b的值
    cpu_cores_num = 6  # CPU核心数
    rhoc_array = []  # 存储结果
    detailed_results = []  # 存储详细结果
    
    for b_para in b_array:
        core_list = np.arange(cpu_cores_num)  # 核心列表
        pool = multiprocessing.Pool()  # 创建进程池

        # 使用部分函数固定参数，包括特殊节点 
        pt = functools.partial(process, b=b_para, nbr_mat=nbr_mat, deg_array=deg_array,
                               nodesnum=nodesnum, lambda_param=lambda_param, special_node=special_node)
        results = pool.map(pt, core_list)  # 多进程并行计算，现在返回元组列表

        # 汇总所有进程的结果
        total_coop_fixations = sum(result[0] for result in results)
        total_absorptions = sum(result[1] for result in results)
        
        # 计算合作固定概率
        rho_c = total_coop_fixations / total_absorptions if total_absorptions > 0 else 0
        rhoc_array.append(rho_c)  # 保存结果
        
        # 保存详细结果
        detailed_results.append({
            "b": b_para,
            "coop_fixations": total_coop_fixations,
            "total_absorptions": total_absorptions,
            "rho_c": rho_c
        })
        
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程结束
        
        print(f"b={b_para}, 合作固定次数={total_coop_fixations}, 总吸收次数={total_absorptions}, rho_c={rho_c}")

    # 保存结果为CSV
    result_df = pd.DataFrame({
        "收益b值": b_array,
        "合作固定概率rho_c": rhoc_array
    })
    csv_filename = f"ba_n{n}_lambda{lambda_param}_single_neighbor_special_node_1.csv"
    result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    
    # 保存详细结果
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_filename = f"detailed_ba_n{n}_lambda{lambda_param}_single_neighbor_special_node.csv"
    detailed_df.to_csv(detailed_csv_filename, index=False, encoding="utf-8-sig")
    
    print(f"简要结果已保存至: {csv_filename}")
    print(f"详细结果已保存至: {detailed_csv_filename}")