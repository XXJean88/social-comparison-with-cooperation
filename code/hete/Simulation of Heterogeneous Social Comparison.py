import networkx as nx
import random
import numpy as np
from numba import jit
import multiprocessing
import functools
import math
import warnings
warnings.filterwarnings('ignore')

# 读取lambda值文件
def load_lambda_values(filename):
    """从文件中读取每个节点的lambda值"""
    lambda_values = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过注释行和空行
                parts = line.split()
                if len(parts) >= 2:
                    node_idx = int(parts[0])
                    lambda_val = float(parts[1])
                    # 确保lambda值按节点顺序存储
                    while len(lambda_values) <= node_idx:
                        lambda_values.append(0.0)
                    lambda_values[node_idx] = lambda_val
    return np.array(lambda_values)

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
def calculate_fitness(payoff_array, nbr_mat, deg_array, node_idx, w, lambda_array):
    """
    计算适应度: F_i(x) = exp(w * (u_i(x) + λ_i * sum(p_im * u_m(x))))
    异质性λ：每个节点使用自己的λ参数
    """
    # 节点自身的收益
    self_payoff = payoff_array[node_idx]
    
    # 获取邻居
    nbrs_num = deg_array[node_idx]  # 邻居数量
    nbrs_array = nbr_mat[node_idx][:nbrs_num]  # 邻居列表
    
    # 计算邻居的平均收益
    neighbors_avg_payoff = np.sum(payoff_array[nbrs_array]) / nbrs_num if nbrs_num > 0 else 0
    
    # 获取该节点的lambda值
    lambda_param = lambda_array[node_idx]
    
    # 计算适应度（包含邻居收益）
    fitness = math.exp(w * (self_payoff + lambda_param * neighbors_avg_payoff))
    
    return fitness

@jit(nopython=True)
def strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_array):
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
        fitness_array[i] = calculate_fitness(payoff_array, nbr_mat, deg_array, neighbor_idx, w, lambda_array)
    
    # 根据适应度选择策略
    total_fitness = np.sum(fitness_array)  # 总适应度
    if total_fitness > 0:
        prob_array = fitness_array / total_fitness  # 计算选择概率
        chosen_neighbor = rand_pick_list(nbrs_array, prob_array)  # 根据概率选择邻居
        state_array[update_node] = state_array[chosen_neighbor]  # 更新策略
    
    return state_array

@jit(nopython=True)
def evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_array):
    total_generation = int(1e9)  # 总演化代数
    payoff_array = np.zeros(nodesnum, dtype=np.float64)  # 收益数组
    state_array = np.zeros(nodesnum, dtype=np.int_)  # 状态数组（0:背叛, 1:合作）
    coop_ini = np.random.choice(nodesnum)  # 随机选择一个初始合作者
    state_array[coop_ini] = 1  # 设置初始合作者
    
    for time in range(total_generation):
        # 进行单轮博弈和策略更新
        payoff_array = single_round(state_array, payoff_array, game_matrix, nbr_mat, deg_array)
        state_array = strategy_update(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w, lambda_array)
        payoff_array[:] = 0  # 重置收益数组
        
        coord = np.sum(state_array)  # 计算合作者数量
        
        # 检查是否达到吸收状态
        if coord > nodesnum - 1:  # 所有节点都合作
            return 1
        if coord == 0:  # 所有节点都背叛
            return 0
            
    return coord / nodesnum  # 返回合作者比例

@jit(nopython=True)
def process(core, b, nbr_mat, deg_array, nodesnum, lambda_array):
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
        freq_c = evolution(game_matrix, nbr_mat, deg_array, nodesnum, w, lambda_array)
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

def compute_rho_c_for_b_value(b, nbr_mat, deg_array, nodesnum, lambda_array, cache_dict):
    """计算单个b值下的rho_c，使用缓存"""
    # 检查缓存
    if b in cache_dict:
        return cache_dict[b]
    
    cpu_cores_num = 7  # CPU核心数
    core_list = np.arange(cpu_cores_num)  # 核心列表
    
    # 使用部分函数固定参数
    pt = functools.partial(process, b=b, nbr_mat=nbr_mat, deg_array=deg_array,
                           nodesnum=nodesnum, lambda_array=lambda_array)
    
    # 多进程并行计算
    with multiprocessing.Pool(cpu_cores_num) as pool:
        results = pool.map(pt, core_list)
    
    # 汇总所有进程的结果
    total_coop_fixations = sum(result[0] for result in results)
    total_absorptions = sum(result[1] for result in results)
    
    # 计算合作固定概率
    rho_c = total_coop_fixations / total_absorptions if total_absorptions > 0 else 0
    
    # 存入缓存
    cache_dict[b] = rho_c
    
    return rho_c

def find_b_for_rho_c_0_2(lambda_filename, network_name="BA"):
    """找到使rho_c最接近0.2的b值"""
    print(f"处理lambda分布: {lambda_filename}")
    
    # 生成n=50的图
    n = 50
    # G = nx.barabasi_albert_graph(n=n, m=2, seed=42)
    # G = nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=42)
    G = nx.random_graphs.random_regular_graph(4, n, seed=42)
    
    # 提取网络信息
    nbrs_dict = nx.to_dict_of_lists(G)  # 邻居字典
    nbr_mat, deg_array = nbr_dict_mat(nbrs_dict)  # 转换为矩阵形式
    nodesnum = G.number_of_nodes()  # 节点数量

    # 加载每个节点的lambda值
    lambda_array = load_lambda_values(lambda_filename)
    
    # 验证lambda数组长度与节点数一致
    if len(lambda_array) != nodesnum:
        print(f"警告: lambda值数量({len(lambda_array)})与节点数({nodesnum})不一致!")
        if len(lambda_array) < nodesnum:
            lambda_array = np.concatenate([lambda_array, np.zeros(nodesnum - len(lambda_array))])
        else:
            lambda_array = lambda_array[:nodesnum]
    
    print(f"已加载{len(lambda_array)}个节点的lambda值")
    
    # 设置b值范围：4.2到4.3，步长为0.05
    b_start = 5.35
    b_end = 5.45
    b_step = 0.05                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    
    # 生成b值数组uni
    b_values = np.arange(b_start, b_end + b_step/2, b_step)
    
    # 计算每个b值的rho_c
    cache_dict = {}
    best_b = None
    best_rho_c = None
    min_diff = float('inf')
    
    print(f"计算b值从{b_start}到{b_end}，步长{b_step}")
    
    for b in b_values:
        rho_c = compute_rho_c_for_b_value(b, nbr_mat, deg_array, nodesnum, lambda_array, cache_dict)
        diff = abs(rho_c - 0.02)
        
        print(f"  b={b:.4f}: rho_c={rho_c:.6f}, 与0.02的差值={diff:.6f}")
        
        if diff < min_diff:
            min_diff = diff
            best_b = b
            best_rho_c = rho_c
    
    # 如果有跨越0.2的两个点，进行更精确的搜索
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
    
    # 如果在两个b值之间跨越了0.2，进行线性插值
    if b_below is not None and b_above is not None:
        print(f"\n在b={b_below:.4f}和b={b_above:.4f}之间跨越了0.2")
        print(f"  b={b_below:.4f}时, rho_c={rho_c_below:.6f}")
        print(f"  b={b_above:.4f}时, rho_c={rho_c_above:.6f}")
        
        # 线性插值
        t = (0.2 - rho_c_below) / (rho_c_above - rho_c_below)
        interpolated_b = b_below + t * (b_above - b_below)
        print(f"  线性插值得到的b值: {interpolated_b:.6f}")
        
        # 计算插值点的rho_c来验证
        interpolated_rho_c = compute_rho_c_for_b_value(interpolated_b, nbr_mat, deg_array, nodesnum, lambda_array, cache_dict)
        print(f"  插值点b={interpolated_b:.6f}的rho_c: {interpolated_rho_c:.6f}")
        
        # 更新最佳值
        interpolated_diff = abs(interpolated_rho_c - 0.2)
        if interpolated_diff < min_diff:
            best_b = interpolated_b
            best_rho_c = interpolated_rho_c
            min_diff = interpolated_diff
    
    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"使rho_c最接近0.2的b值: {best_b:.6f}")
    print(f"对应的rho_c值: {best_rho_c:.6f}")
    print(f"与0.2的差值: {min_diff:.6f}")
    print(f"{'='*60}\n")
    
    return best_b

if __name__ == "__main__":
    # lambda分布文件列表
    lambda_files = [
        # "lambda_powerlaw_rg_n50.txt",
        # "lambda_exponential_rg_n50.txt",
        "lambda_uniform_rg_n50.txt",
        # "lambda_normal_rg_n50.txt"
    ]
    
    # 对每种lambda分布运行计算
    results = {}
    for lambda_file in lambda_files:
        try:
            print(f"\n{'='*80}")
            print(f"开始处理: {lambda_file}")
            b_value = find_b_for_rho_c_0_2(lambda_file)
            results[lambda_file] = b_value
        except Exception as e:
            print(f"处理文件 {lambda_file} 时出错: {str(e)}")
    
    # 输出所有结果
    print(f"\n{'='*80}")
    print("所有lambda分布的结果汇总:")
    for lambda_file, b_value in results.items():
        dist_name = lambda_file.replace("lambda_", "").replace(".txt", "").replace("_ba_n50", "")
        print(f"{dist_name}: b = {b_value:.6f}")
    print(f"{'='*80}")