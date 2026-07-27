########## Here are the associated DomiRank functions #############
import time
import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx

########## Here are the general functions needed for efficient dismantling and testing of networks #############

def get_largest_component(G, strong = False):
    '''
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph'''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return G

##########################方案一###################################################################


def relabel_nodes(G, yield_map=False, return_reverse=False):
    # 获取排序后的节点列表确保一致性
    sorted_nodes = sorted(G.nodes())
    mapping = dict(zip(sorted_nodes, range(len(G))))
    
    # 创建反向映射
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    G_relabeled = nx.relabel_nodes(G, mapping)
    
    if not yield_map:
        return G_relabeled
    else:
        # 修复返回值中的关键字匹配问题
        return G_relabeled, mapping, reverse_mapping
def relabel_nodes1(G, yield_map = False):
    '''relabels the nodes to be from 0, ... len(G).
    1. Yield_map returns an extra output as a dict. in case you want to save the hash-map to retrieve node-id'''
    if yield_map == True:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G    
def get_component_size(G, strong = False):
    '''
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph'''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
        return len(GMask)        
    elif type(G) == scipy.sparse.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        noComponent, lenComponent = sp.sparse.csgraph.connected_components(G, directed = True, connection = connection_type, return_labels = True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')
        
def get_link_size(G):
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        links = len(G.edges()) #convert to scipy sparse if it is a graph 
    elif type(G) == scipy.sparse.csr_array:
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links

def remove_node(G, removedNode):
    '''
    removes the node from the graph by removing it from a networkx.Graph type, or zeroing the edges in array form.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if type(removedNode) == int:
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node) #remove node in graph form
        return G
    elif type(G) == scipy.sparse.csr_array:
        diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0])) 
        diag[removedNode, removedNode] = 0 #set the rows and columns that are equal to zero in the sparse array
        G = diag @ G 
        return G @ diag
    
def generate_attack(centrality, node_map = False):
    '''we generate an attack based on a centrality measure - 
    you can possibly input the node_map to convert the attack to have the correct nodeID'''
    if node_map == False:
        node_map = range(len(centrality))
    else:
        node_map = list(node_map.values())
    zipped = dict(zip(node_map, centrality))
    attackStrategy = sorted(zipped, reverse = True, key = zipped.get)
    return attackStrategy


import numpy as np

def generate_recovery(centrality, node_map=False):
    """
    生成随机恢复序列（接口与generate_attack完全一致）
    
    参数:
        centrality: list或np.array - 节点的中心性值（实际在随机恢复中不会使用）
        node_map: dict或False - 节点ID映射字典（与攻击策略相同逻辑）
    
    返回:
        list: 随机顺序的节点ID列表（若node_map=False则为原始索引，否则为映射后的ID）
    """
    # Step 1: 处理node_map（完全复制攻击策略的逻辑）
    if node_map == False:
        node_ids = list(range(len(centrality)))  # 默认使用连续整数索引
    else:
        node_ids = list(node_map.values())       # 使用映射后的实际节点ID
    
    # Step 2: 随机打乱节点顺序（无视centrality）
    np.random.shuffle(node_ids)  # 就地打乱
    
    return node_ids


def network_attack_sampled(G, attackStrategy, sampling=0):
    '''
    增强版网络攻击模拟：记录达到指定LCC比例(默认50%)所需的节点移除比例
    新增返回：critical_ratio（使LCC降至目标的节点移除比例）
    '''
    if type(G) == nx.classes.graph.Graph:
        GAdj = nx.to_scipy_sparse_array(G, weight="weight")
    else:
        GAdj = G.copy()
    
    N = GAdj.shape[0]
    if sampling == 0:
        sampling = 1 if N < 100 else int(N/100)
    target_lcc_ratio=0.5
    initialComponent = get_component_size(GAdj)
    initialLinks = get_link_size(GAdj)
    
    # 初始化记录数组 (+1 用于存储初始状态)
    num_points = (N-1)//sampling + 2
    componentEvolution = np.zeros(num_points)
    linksEvolution = np.zeros(num_points)
    removed_ratios = np.zeros(num_points)  # 新增：节点移除比例
    
    # 初始状态记录
    componentEvolution[0] = 1.0  # 相对初始LCC
    linksEvolution[0] = 1.0      # 相对初始链接数
    removed_ratios[0] = 0.0      # 初始状态无移除
    j = 1
    
    # 关键指标跟踪
    critical_ratio = None         # 达到目标的结点移除比
    
    for i in range(1, N):
        if i % sampling == 0 or i == N-1:
            # 移除节点批次：前i个节点
            nodes_to_remove = attackStrategy[max(0, i-sampling):i]
            GAdj = remove_node(GAdj, nodes_to_remove)
            
            # 计算当前状态
            comp_ratio = get_component_size(GAdj) / initialComponent
            link_ratio = get_link_size(GAdj) / initialLinks
            removal_ratio = i / N  # 当前节点移除比例
            
            # 记录状态
            componentEvolution[j] = comp_ratio
            linksEvolution[j] = link_ratio
            removed_ratios[j] = removal_ratio
            
            # 检测是否达到目标LCC比例 (首次到达)
            if critical_ratio is None and comp_ratio <= target_lcc_ratio:
                critical_ratio = removal_ratio
                critical_idx = j
            
            j += 1
            
            # 提前终止：若网络已完全瓦解
            if comp_ratio == 0:
                break
    
    # 裁剪结果数组
    componentEvolution = componentEvolution[:j]
    linksEvolution = linksEvolution[:j]
    removed_ratios = removed_ratios[:j]
    
    # 返回关键指标 + 演化曲线
    return {
        "critical_ratio": critical_ratio,  # 关键节点移除比
        "removed_ratios": removed_ratios,   # 移除比例数组
        "components": componentEvolution,      # LCC比例数组
        "links": linksEvolution                 # 链接比例数组
    }


# ######## Beginning of domirank stuff! ####################


#############################################################################方案一，也算是局部贪心###############################################
# def domirank(G, analytical = True, sigma = -1, dt = 0.1, epsilon = 1e-6, maxIter = 1000, checkStep = 10,
#                            krylov_dim=100, krylov_rtol=1e-5, verbose=True):
#     """
#     DomiRank的Krylov子空间近似法。
#     专为在不破坏全局结构的前提下进行高性能近似而设计。

#     Args:
#         krylov_dim (int): Krylov子空间的维度 (迭代次数)。
#                          值越小，速度越快，但近似程度越高。
#                          通常设在 100-500 之间。
#         krylov_tol (float): 求解器收敛的容忍度。

#     """
#     import scipy.sparse as sp
#     from scipy.sparse.linalg import gmres, cg
#     # --- 1. 初始化 ---
#     if isinstance(G, (nx.Graph, nx.DiGraph)):
#         G = nx.to_scipy_sparse_array(G, weight="weight", format='csr')
#     else:
#         G = G.copy()

#     N = G.shape[0]
#     if N == 0: return True, np.array([])
    
#     if analytical:
#         return False, np.zeros(N)

#     if sigma == -1:
#         if verbose: print("计算最优sigma...")
#         sigma = optimal_sigma(G, analytical=True)

#     # --- 2. 构造线性系统 BΓ = v ---
#     if verbose: print("构造线性系统...")
#     I = sp.identity(N, format='csr')
#     B = I + sigma * G
#     v = sigma * G.sum(axis=1)
    
#     if not isinstance(v, np.ndarray):
#         v = v.A.flatten()
    
#     # 检查矩阵 B 是否对称
#     is_symmetric = (B - B.T).nnz == 0
    
#     # 创建一个回调函数来监控迭代过程
#     class IterationCounter:
#         def __init__(self, verbose_step=20):
#             self.count = 0
#             self.verbose_step = verbose_step
#         def __call__(self, xk):
#             self.count += 1
#             if verbose and self.count % self.verbose_step == 0:
#                 print(f"  Krylov Iter {self.count}...")
    
#     callback = IterationCounter()

#     print(f"--- 开始使用Krylov子空间法求解 (子空间维度上限: {krylov_dim}, 容忍度: {krylov_rtol:.1e}) ---")
#     start_time = time.time()
    
#     if is_symmetric:
#         print("检测到对称矩阵，使用共轭梯度法 (CG)...")
#         try:
#             # *** 关键修正：使用 rtol 参数 ***
#             Psi, exit_code = cg(B, v, rtol=krylov_rtol, maxiter=krylov_dim, callback=callback)
            
#             if exit_code == 0:
#                 print(f"CG成功收敛于 {callback.count} 次迭代。")
#             elif exit_code > 0:
#                 print(f"警告: CG在 {callback.count} 次迭代后未达到收敛容忍度 (这是正常的近似行为)。")
#             else:
#                 print(f"错误: CG计算失败，exit_code={exit_code}")
#                 return False, np.zeros(N)
#         except Exception as e:
#             print(f"CG 求解失败: {e}")
#             return False, np.zeros(N)
#     else:
#         print("检测到非对称矩阵，使用广义最小残差法 (GMRES)...")
#         try:
#             # GMRES的回调函数参数是残差范数，与CG不同，所以我们用一个简化的回调
#             # 或者干脆不用回调，因为GMRES的迭代次数由exit_code直接给出
#             Psi, exit_code = gmres(B, v, tol=krylov_rtol, maxiter=krylov_dim)
            
#             if exit_code == 0:
#                 print(f"GMRES成功收敛。")
#             elif exit_code > 0:
#                 # 在GMRES中, exit_code > 0 直接表示迭代次数
#                 print(f"警告: GMRES在 {exit_code} 次迭代后未达到收敛容忍度 (这是正常的近似行为)。")
#             else:
#                 print(f"错误: GMRES计算失败，exit_code={exit_code}")
#                 return False, np.zeros(N)
#         except Exception as e:
#             print(f"GMRES 求解失败: {e}")
#             return False, np.zeros(N)
    
#     end_time = time.time()
#     print(f"Krylov求解完成，耗时: {end_time - start_time:.4f}s")
    
#     return True, Psi
##############################################################################损失精度的冻结方法#############################################




# def lis_pruning(G, freeze_quantile=0.2):
#     """
#     基于“局部影响力得分 (LIS)”的静态剪枝。
#     识别那些出度低且邻居环境简单的节点进行冻结。
    
#     *** 版本 2: 兼容有向图和无向图 ***
#     """
#     import scipy.sparse as sp
#     import numpy as np
    
#     print("--- 开始基于“局部影响力得分 (LIS)”的静态剪枝 (V2) ---")
#     N = G.shape[0]
#     if N == 0: return np.array([], dtype=bool)

#     if (G != G.T).nnz == 0:
#         print("检测到无向图 (邻接矩阵对称)")
#         is_undirected = True
#     else:
#         print("检测到有向图 (邻接矩阵非对称)")
#         is_undirected = False

#     out_degrees = np.asarray(G.sum(axis=1)).flatten()
#     in_degrees = np.asarray(G.sum(axis=0)).flatten()
    
#     neighbor_in_degree_sum = G @ in_degrees
    
#     # 对于无向图，out_degrees 就是节点的度数。
#     lis_scores = np.log1p(out_degrees) * np.log1p(neighbor_in_degree_sum)
#     if freeze_quantile <= 0:
#         return np.ones(N, dtype=bool)
#     if freeze_quantile >= 1:
#         return np.zeros(N, dtype=bool)

#     threshold = np.quantile(lis_scores, freeze_quantile)
    
#     if np.all(lis_scores == lis_scores[0]):
#          print("警告: 所有节点的LIS分数均相同，无法通过阈值区分。不执行剪枝。")
#          return np.ones(N, dtype=bool)

#     active_mask = lis_scores > threshold
    
#     num_active = np.sum(active_mask)
#     num_frozen = N - num_active
#     target_frozen = int(N * freeze_quantile)
    
#     print(f"LIS剪枝完成。冻结阈值: {threshold:.4e}。")
#     print(f"目标冻结 {target_frozen} 个节点，实际冻结 {num_frozen} 个。")
#     print(f"活跃节点: {num_active}, 冻结节点: {num_frozen}")
    
#     return active_mask


# import scipy.sparse as sp
# import networkx as nx

# import numpy as np


# def sfs_pruning(G, freeze_quantile=None):
#     """
#     基于“k-核分解”的静态剪枝 

#     此版本采用图论中成熟的k-核分解理论来精确划分网络的核心与外围，
#     旨在找到一个数量可观且足够安全的冻结节点集。

#     核心思想:
#     1. 在自动模式下，冻结所有“弱依赖节点”。
#     2. “弱依赖节点”的精确定义: 在网络的无向骨架上，核数(core number)为1的节点。
#        这些节点构成了附着在核心区域(所有环路的所在地)之外的树状和链状结构。
#     3. 手动模式下，将核数作为主要的排序标准，核数越低越优先冻结。

#     Args:
#         G (sp.spmatrix): 原始的稀疏邻接矩阵 (可以是有向或无向)。
#         freeze_quantile (float or None): 冻结比例或自动模式开关。

#     Returns:
#         np.array (dtype=bool): 活跃节点掩码 (True表示核心, False表示外围)。
#     """
#     N = G.shape[0]
#     if N == 0:
#         return np.array([], dtype=bool)

#     print("--- 开始基于“k-核分解”的静态剪枝 ---")

#     # --- 步骤1: 预计算所有节点的核数 ---
#     print("正在进行k-核分解以计算所有节点的核数...")
    
#     A = G + G.T
#     A.setdiag(0)
#     A.eliminate_zeros()

#     G_nx = nx.from_scipy_sparse_array(A)
    
#     core_numbers_dict = nx.core_number(G_nx)

#     core_numbers = np.array([core_numbers_dict.get(i, 0) for i in range(N)])
#     print("核数计算完成。")
    
#     # --- 步骤2: 根据模式执行剪枝 ---
#     if freeze_quantile is None:
#         print("--- 自动模式: 正在冻结所有核数为1的“弱依赖节点” ---")
        
#         indices_to_freeze = np.where(core_numbers == 1)[0]
        
#         if len(indices_to_freeze) > 0.8 * N:
#             print(f"警告：核数为1的节点过多({len(indices_to_freeze)}), 将冻结比例限制在80%。")
#             # 在手动模式下，按度数排序来决定冻结哪些
#             total_degree = np.array(A.sum(axis=1)).flatten()
#             peripheral_scores = total_degree[indices_to_freeze]
#             num_to_freeze = int(0.8 * N)
#             # 找到度数最低的那些核数为1的节点
#             cutoff_indices = np.argpartition(peripheral_scores, num_to_freeze -1)[:num_to_freeze]
#             indices_to_freeze = indices_to_freeze[cutoff_indices]

#         active_mask = np.ones(N, dtype=bool)
#         if len(indices_to_freeze) > 0:
#             active_mask[indices_to_freeze] = False
        
#         print(f"“k-核”剪枝完成。自动识别并冻结了 {len(indices_to_freeze)} 个节点。")
#     else:
#         raise ValueError("freeze_quantile 必须是 [0, 1] 范围内的浮点数或 None。")

#     num_active = np.sum(active_mask)
#     print(f"最终结果 -> 活跃节点: {num_active}, 冻结节点: {N - num_active}")
    
#     return active_mask

# def domirank(G, analytical=True, sigma=-1, dt=0.1, epsilon=1e-6, maxIter=1000, checkStep=10, use_freezing=True, freeze_quantile=None, _cache={}):
#     """
#     DomiRank
#     """
#     # 1. 统一输入和解析解快速通道
#     if isinstance(G, nx.classes.graph.Graph):
#         G = nx.to_scipy_sparse_array(G, weight="weight")
#     else:
#         G = G.copy() # 确保不修改原始对象
#     import scipy.sparse as sp
#     N = G.shape[0]
#     if analytical:
#         # 解析解分支保持不变，它已经非常快
#         if sigma == -1:
#             sigma = optimal_sigma(G, analytical=True)
#         identity = sp.sparse.identity(N)
#         b_vector = sigma * G.sum(axis=1) # b_vector应为(N,1)或(N,)
#         Psi = sp.sparse.linalg.spsolve(sigma * G + identity, b_vector)
#         print("使用解析解法")
#         return True, Psi


#     graph_hash = (G.data.tobytes(), G.indices.tobytes(), G.indptr.tobytes()) # 更可靠的哈希
#     if use_freezing:
#         if graph_hash in _cache and 'active_mask' in _cache[graph_hash]:
#             active_mask = _cache[graph_hash]['active_mask']
#         else:
#             active_mask = lis_pruning(G, freeze_quantile=0.15)
#             if graph_hash not in _cache:
#                 _cache[graph_hash] = {}
#             _cache[graph_hash]['active_mask'] = active_mask
#     else:
#         active_mask = np.ones(N, dtype=bool)

#     active_indices = np.where(active_mask)[0]
#     num_active = len(active_indices)

#     if sigma == -1:
#         sigma, _ = optimal_sigma(G, analytical=False)
    
#     pGAdj = (sigma * G).astype(np.float64)
#     Psi = np.ones(N, dtype=np.float64) / N # 初始Psi
#     frozen_indices = np.where(~active_mask)[0]
#     pG_aa = pGAdj[active_indices, :][:, active_indices].tocsr()
#     pG_af = pGAdj[active_indices, :][:, frozen_indices].tocsr()
#     pG_fa = pGAdj[frozen_indices, :][:, active_indices].tocsr()

#     Psi_a = Psi[active_indices]
#     Psi_f = Psi[frozen_indices]

#     influence_from_frozen = pG_af @ (1 - Psi_f)
#     row_sums_aa = np.array(pG_aa.sum(axis=1)).flatten()
#     row_sums_fa = np.array(pG_fa.sum(axis=1)).flatten()
    
#     active_boundary = epsilon * num_active * dt
#     maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)
#     j = 0
#     for i in range(maxIter):
#         influence_from_active = row_sums_aa - (pG_aa @ Psi_a)
        
#         term1_a = influence_from_active + influence_from_frozen
#         tempVal_a = (term1_a - Psi_a) * dt
#         Psi_a += tempVal_a

#         if i % checkStep == 0:
#             if np.abs(tempVal_a).sum() < active_boundary:
#                 break
#             maxVals[j] = tempVal_a.max()
#             if j > 1 and maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
#                 return False, Psi # 返回部分计算的结果
#             j += 1
#     support_from_active = row_sums_fa - (pG_fa @ Psi_a)
#     final_frozen_scores = support_from_active * dt 
    
#     Psi_final = np.empty(N, dtype=np.float64)
#     Psi_final[active_indices] = Psi_a
#     Psi_final[frozen_indices] = final_frozen_scores

#     return True, Psi_final


#######################################################压缩图方法###################################################
import numpy as np
import scipy.sparse as sp
import networkx as nx

# --- 模块级缓存，用于存储预计算结果 ---
_COMPRESSION_CACHE = {}
import hashlib
from collections import defaultdict

# def find_structurally_congruent_vertices(G: sp.spmatrix):
#     """
#     根据精确的拓扑和权重定义，寻找图中所有“结构全等”的节点集。
#     (优化版：通过度数剪枝预先过滤掉大量不可能是全等的节点)
#     """
#     print("--- 正在寻找“结构全等”节点集 (度数剪枝优化版)... ---")
#     import scipy.sparse as sp
#     # --- 1. 初始化和准备工作 ---
#     N = G.shape[0]
#     if N == 0:
#         return [], []
        
#     G_csr = G.tocsr()
#     is_directed = (G != G.T).nnz > 0
    
#     # --- 2. 优化核心：度数预计算和初始分区 ---
#     # 这是最关键的优化步骤。如果两个节点全等，它们的(入/出)度数必须相同。
#     # 我们可以先按度数对所有节点分组，从而避免对度数不同的节点进行比较。
    
#     # 高效计算所有节点的出度
#     out_degrees = G_csr.indptr[1:] - G_csr.indptr[:-1]
    
#     partitions = defaultdict(list)
#     if is_directed:
#         G_csc = G.tocsc()
#         # 高效计算所有节点的入度
#         in_degrees = G_csc.indptr[1:] - G_csc.indptr[:-1]
#         for i in range(N):
#             # 使用 (出度, 入度) 元组作为分组的键
#             partitions[(out_degrees[i], in_degrees[i])].append(i)
#     else:
#         G_csc = None # 明确设为None，后面会用到
#         for i in range(N):
#             # 使用度数作为分组的键
#             partitions[out_degrees[i]].append(i)

#     # --- 3. 剪枝 (Pruning) ---
#     # 遍历分区，如果一个分区内只有一个节点，那么这个节点是度数唯一的，
#     # 它绝不可能有全等节点。我们直接将其放入唯一节点列表，并从后续计算中排除。
    
#     unique_nodes = []
#     candidate_groups = []
#     for group in partitions.values():
#         if len(group) == 1:
#             unique_nodes.extend(group)
#         else:
#             # 只有节点数 > 1 的分组才可能是全等集，加入候选列表
#             candidate_groups.append(group)
            
#     print(f"度数剪枝完成。找到 {len(unique_nodes)} 个度数唯一的节点。")
#     print(f"剩下 {len(candidate_groups)} 个候选组 (共 {N - len(unique_nodes)} 个节点) 需要进一步计算签名。")

#     # --- 4. 在候选分区内进行精细签名计算 ---
#     # 现在，我们只对那些无法通过度数区分的“候选组”进行昂贵的签名计算。
    
#     congruent_sets = []
    
#     # 辅助函数，用于计算单个节点的完整签名（代码与原版相同，封装起来更清晰）
#     def _calculate_signature(node_idx, G_csr_ref, G_csc_ref, is_directed_ref):
#         if is_directed_ref:
#             # 出边签名
#             out_indices = G_csr_ref.indices[G_csr_ref.indptr[node_idx]:G_csr_ref.indptr[node_idx+1]]
#             out_weights = G_csr_ref.data[G_csr_ref.indptr[node_idx]:G_csr_ref.indptr[node_idx+1]]
#             sort_idx_out = np.argsort(out_indices)
#             out_sig = out_indices[sort_idx_out].tobytes() + out_weights[sort_idx_out].tobytes()

#             # 入边签名
#             in_indices = G_csc_ref.indices[G_csc_ref.indptr[node_idx]:G_csc_ref.indptr[node_idx+1]]
#             in_weights = G_csc_ref.data[G_csc_ref.indptr[node_idx]:G_csc_ref.indptr[node_idx+1]]
#             sort_idx_in = np.argsort(in_indices)
#             in_sig = in_indices[sort_idx_in].tobytes() + in_weights[sort_idx_in].tobytes()
            
#             return hashlib.sha256(out_sig + in_sig).digest()
#         else:
#             neighbors = G_csr_ref.indices[G_csr_ref.indptr[node_idx]:G_csr_ref.indptr[node_idx+1]]
#             weights = G_csr_ref.data[G_csr_ref.indptr[node_idx]:G_csr_ref.indptr[node_idx+1]]
#             sort_idx = np.argsort(neighbors)
#             sig = neighbors[sort_idx].tobytes() + weights[sort_idx].tobytes()
#             return hashlib.sha256(sig).digest()

#     # 遍历每个候选组
#     for group in candidate_groups:
#         # 在组内根据完整签名再次进行分组
#         signatures_in_group = defaultdict(list)
#         for node_idx in group:
#             sig = _calculate_signature(node_idx, G_csr, G_csc, is_directed)
#             signatures_in_group[sig].append(node_idx)
            
#         # --- 5. 构建最终结果 ---
#         # 遍历组内细分后的结果
#         for final_group in signatures_in_group.values():
#             if len(final_group) > 1:
#                 C = np.array(final_group)
                
#                 # 对于无向图，检查全等集内部是否有连接
#                 if not is_directed:
#                     if G[C, :][:, C].nnz > 0:
#                         # 如果内部有连接，则此集合不符合定义，其成员都视为独特
#                         unique_nodes.extend(C)
#                         continue

#                 congruent_sets.append(list(C))
#             else:
#                 # 签名计算后，发现这个节点在组内也是唯一的
#                 unique_nodes.extend(final_group)

#     print(f"查找完成。共找到 {len(congruent_sets)} 个全等节点集和 {len(unique_nodes)} 个唯一节点。")
#     return congruent_sets, unique_nodes

def find_structurally_congruent_vertices(G: sp.spmatrix):
    """
    根据“输出全等性”寻找可压缩的节点集。
    这是一种更宽松的规则，旨在为非对称压缩模型找到更多可压缩节点。
    节点被认为是“输出全等”的，如果它们有完全相同的出边邻居和权重。
    """
    print("--- 正在寻找“输出全等”节点集 (宽松规则版)... ---")
    
    # --- 1. 初始化 ---
    N = G.shape[0]
    if N == 0:
        return [], []
        
    G_csr = G.tocsr()
    is_directed = (G != G.T).nnz > 0
    # 在这个模型下，我们主要关心有向图的特性，但代码对无向图同样有效。

    # --- 2. 核心修改(1): 只根据出度进行初始分区 ---
    out_degrees = G_csr.indptr[1:] - G_csr.indptr[:-1]
    
    partitions = defaultdict(list)
    for i in range(N):
        # 只使用出度作为分组的键
        partitions[out_degrees[i]].append(i)

    # --- 3. 剪枝 (与之前相同) ---
    unique_nodes = []
    candidate_groups = []
    for group in partitions.values():
        if len(group) == 1:
            unique_nodes.extend(group)
        else:
            candidate_groups.append(group)
            
    print(f"出度剪枝完成。找到 {len(unique_nodes)} 个出度唯一的节点。")
    print(f"剩下 {len(candidate_groups)} 个候选组 (共 {N - len(unique_nodes)} 个节点) 需要进一步计算签名。")

    # --- 4. 核心修改(2): 只根据出边信息计算签名 ---
    congruent_sets = []
    
    def _calculate_output_signature(node_idx, G_csr_ref):
        """只计算出边签名"""
        out_indices = G_csr_ref.indices[G_csr_ref.indptr[node_idx]:G_csr_ref.indptr[node_idx+1]]
        out_weights = G_csr_ref.data[G_csr_ref.indptr[node_idx]:G_csr_ref.indptr[node_idx+1]]
        
        # 排序以保证顺序无关性
        sort_idx_out = np.argsort(out_indices)
        
        out_sig_bytes = out_indices[sort_idx_out].tobytes() + out_weights[sort_idx_out].tobytes()
        
        return hashlib.sha256(out_sig_bytes).digest()

    for group in candidate_groups:
        signatures_in_group = defaultdict(list)
        for node_idx in group:
            # 只调用新的、更简单的签名函数
            sig = _calculate_output_signature(node_idx, G_csr)
            signatures_in_group[sig].append(node_idx)
            
        # --- 5. 构建最终结果 (与之前相同) ---
        for final_group in signatures_in_group.values():
            if len(final_group) > 1:
                C = np.array(final_group)
                
                # 对于无向图的特殊检查依然保留，以防万一
                if not is_directed:
                    if G[C, :][:, C].nnz > 0:
                        unique_nodes.extend(C)
                        continue

                congruent_sets.append(list(C))
            else:
                unique_nodes.extend(final_group)

    print(f"查找完成。共找到 {len(congruent_sets)} 个输出全等节点集和 {len(unique_nodes)} 个唯一节点。")
    return congruent_sets, unique_nodes

def precompute_compression_data(G: sp.spmatrix):
    """
    使用高效的“混合构建法”为图执行特殊的“非对称”压缩。
    - 超节点的出边(行): 等于其代表元节点的出边。
    - 超节点的入边(列): 等于其所有成员节点入边之和。
    此方法在保证模型正确性的前提下，性能远超原始的 bmat 方法。
    """
    global _COMPRESSION_CACHE
    import scipy.sparse as sp
    graph_hash = (G.data.tobytes(), G.indices.tobytes(), G.indptr.tobytes())
    if graph_hash in _COMPRESSION_CACHE:
        return _COMPRESSION_CACHE[graph_hash]

    print("--- 首次执行非对称图压缩 (混合构建优化版) ---")

    congruent_sets, unique_nodes = find_structurally_congruent_vertices(G)

    if not congruent_sets:
        plan = {'is_compressible': False}
        _COMPRESSION_CACHE[graph_hash] = plan
        return plan

    N = G.shape[0]
    num_unique = len(unique_nodes)
    num_supernodes = len(congruent_sets)
    N_comp = num_unique + num_supernodes
    representatives = [s[0] for s in congruent_sets]
    
    print(f"正在使用混合构建法构建非对称压缩图 (大小: {N_comp}x{N_comp})...")

    # --- 1. 行选择：提取所有相关的出边 ---
    # 我们只关心唯一节点和代表元节点的出边。
    source_rows_indices = np.concatenate([unique_nodes, representatives])
    
    # 通过一次高效的切片操作，创建只包含这些行的子图。
    G_sub = G[source_rows_indices, :]

    # --- 2. 列聚合：对子图的列进行重映射和求和 ---
    # a) 创建一个将“原图任意列”映射到“压缩图新列”的查找表
    node_to_new_col_idx = np.empty(N, dtype=np.int32)
    unique_nodes_arr = np.array(unique_nodes)
    node_to_new_col_idx[unique_nodes_arr] = np.arange(num_unique)
    for i, s in enumerate(congruent_sets):
        supernode_idx = num_unique + i
        node_to_new_col_idx[s] = supernode_idx
        
    # b) 将子图转换为COO格式，以方便地访问其列索引
    G_sub_coo = G_sub.tocoo()
    
    # c) 对列进行重映射。行索引保持不变，因为它们已经是正确的出边来源了。
    # G_sub_coo.row 的索引是相对于 G_sub 的，范围是 [0, N_comp-1]，这正好是我们最终需要的行索引。
    final_rows = G_sub_coo.row
    final_cols = node_to_new_col_idx[G_sub_coo.col]
    
    # --- 3. 一步构建：生成最终的压缩图 ---
    # csr_matrix 构造函数会自动将具有相同 (行, 列) 坐标的权重相加，
    # 完美地实现了我们对列的聚合要求。
    G_comp = sp.csr_matrix((G_sub_coo.data, (final_rows, final_cols)), shape=(N_comp, N_comp))
    G_comp.eliminate_zeros()
    
    # 打包 plan
    plan = {
        'is_compressible': True,
        'G_comp': G_comp,
        # ... (可以添加其他需要的信息)
        'unique_nodes': unique_nodes_arr, 
        'congruent_sets': congruent_sets,
        'num_unique': num_unique,
        'num_supernodes': num_supernodes
    }
    
    _COMPRESSION_CACHE[graph_hash] = plan
    return plan





def domirank(G, analytical=False, sigma=-1, dt=0.1, epsilon=1e-6, maxIter=1000, checkStep=10): 

    if isinstance(G, nx.classes.graph.Graph):
        G = nx.to_scipy_sparse_array(G, weight="weight")
    else:
        G = G.copy()
    
    N = G.shape[0]
    
    if analytical:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        Psi = sp.sparse.linalg.spsolve(sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
        return True, Psi

    
    # 1. 获取预计算的压缩计划 (如果已缓存，此步极快)
    compression_plan = precompute_compression_data(G)
    if sigma == -1: sigma, _ = optimal_sigma(G, analytical=False)
    Psi = np.ones(N).astype(np.float64) / N

    

    G_comp = compression_plan['G_comp']
    unique_nodes = compression_plan['unique_nodes']
    congruent_sets = compression_plan['congruent_sets']
    num_unique = compression_plan['num_unique']
    N_comp = G_comp.shape[0]
    

    pGAdj_comp = (sigma * G_comp).astype(np.float64)
    Psi_comp = np.ones(N_comp, dtype=np.float64) / N
    maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64) 
    j = 0
    boundary_comp = epsilon * N_comp * dt
    for i in range(maxIter):
        term1 = pGAdj_comp @ (1 - Psi_comp)
        tempVal = (term1 - Psi_comp) * dt
        Psi_comp += tempVal.real
        if i % checkStep == 0:
            if np.abs(tempVal).sum() < boundary_comp:
                print(f"压缩图在第 {i} 步收敛。")
                break
        
            maxVals[j] = tempVal.max()
            if i == 0:
                initialChange = maxVals[j]
            if j > 0:
                if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                    return False, Psi
            j+=1
      
    Psi_final = np.empty(N, dtype=np.float64)
    Psi_final[unique_nodes] = Psi_comp[:num_unique]
    for i, s in enumerate(congruent_sets):
        supernode_score = Psi_comp[num_unique + i]
        Psi_final[s] = supernode_score
        
    return True, Psi_final


def standard_domirank_with_warm_start(G, sigma, dt, epsilon, maxIter, checkStep):
    N = G.shape[0]
    
    # 在完整图上执行“闪电热启动”
    # try:
    #     d_out = np.array(G.sum(axis=1)).flatten()
    #     d_out_inv = np.zeros_like(d_out)
    #     has_out_edges = d_out > 0
    #     d_out_inv[has_out_edges] = 1.0 / d_out[has_out_edges]
    #     P = sp.diags(d_out_inv) @ G.tocsr()
        
    #     psi_warm = np.ones(N) / N
    #     for _ in range(10):
    #         psi_warm = P.T @ psi_warm
        
    #     Psi = psi_warm / psi_warm.sum()
    # except Exception:
    Psi = np.ones(N).astype(np.float64) / N
        
    # 在完整图上运行标准迭代
    pGAdj = (sigma * G).astype(np.float64)
    row_sums = np.array(pGAdj.sum(axis=1)).flatten()
    boundary = epsilon * N * dt
    
    for i in range(maxIter):
        term1 = row_sums - (pGAdj @ Psi)
        tempVal = (term1 - Psi) * dt
        Psi += tempVal.real
        if i % checkStep == 0 and np.abs(tempVal).sum() < boundary:
            print(f"标准图在第 {i} 步快速收敛。")
            break
    
    return True, Psi

#######################################################迭代冻结方法##################################################



import scipy as sp 
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 100, dt = 0.1, epsilon = 1e-5, maxIter = 1000, checkStep = 10):
    '''
    G: is the input graph as a sparse array.
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm.
    Currently this function is only single-threaded, as the bisection algorithm only allows for single-threaded
    exection. Note, that this algorithm is slightly different, as it uses the fact that DomiRank diverges
    at values larger than -1/lambN to its benefit, and thus, it is not exactly bisection theorem. I haven't
    tested in order to see which exact value is the fastest for execution, but that will be done soon!
    Some notes:
    Increase maxDepth for increased accuracy.
    Increase maxIter if DomiRank doesn't start diverging within 100 iterations -- i.e. increase at the expense of 
    increased computational cost if you want potential increased accuracy.
    Decrease checkstep for increased error-finding for the values of sigma that are too large, but higher compcost
    if you are frequently less than the value (but negligible compcost).
    '''
    x = (minVal + maxVal)/G.sum(axis=-1).max()
    minValStored = 0
    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x
            x = (minVal + maxVal)/2
            minValStored = minVal
        else:
            maxVal = (x + maxVal)/2
            x = (minVal + maxVal)/2
        if minVal == 0:
            print(f'Current Interval : [-inf, -{1/maxVal}]')
        else:
            print(f'Current Interval : [-{1/minVal}, -{1/maxVal}]')
    finalVal = (maxVal + minVal)/2
    return -1/finalVal



############## This section is for finding the optimal sigma #######################

def process_iteration(q, i,analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
    # if isinstance(q, tuple) and len(q) == 4:
    #     data, indices, indptr, shape = q
    #     spArray = sp.csr_matrix((data, indices, indptr), shape=shape)
    # else:
    #     # 如果传入的已经是矩阵对象（非多进程场景），直接使用
    #     spArray = q
    tf, domiDist = domirank(spArray,analytical = analytical, sigma = sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    domiAttack = generate_attack(domiDist)
    result= network_attack_sampled(spArray, domiAttack, sampling = sampling)
    finalErrors =  result['components'].sum()
    q.put((i, finalErrors))

def optimal_sigma(spArray, analytical = True, endVal = 0, startval = 0.000001, iterationNo = 100, dt = 0.1, epsilon = 1e-5, maxIter = 300, checkStep = 10, maxDepth = 100, sampling = 0):
    ''' This part finds the optimal sigma by searching the space, here are the novel parameters:
    spArray: is the input sparse array/matrix for the network.
    startVal: is the starting value of the space that you want to search.
    endVal: is the ending value of the space that you want to search (normally it should be the eigenvalue)
    iterationNo: the number of partitions of the space between lambN that you set
    
    return : the function returns the value of sigma - the numerator of the fraction of (\sigma)/(-1*lambN)
    '''
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth = maxDepth, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep )
    import multiprocessing as mp
    endval = -0.9999/endVal
    tempRange = np.arange(startval, endval + (endval-startval)/iterationNo, (endval-startval)/iterationNo)
    processes = []
    q = mp.Queue()
    for i, sigma in enumerate(tempRange):
        p = mp.Process(target=process_iteration, args=(q, i,analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
        p.start()
        processes.append(p)

    results = [None] * len(tempRange)  # Initialize a results list

    #Join the processes and gather results from the queue
    for p in processes:
        p.join()

    #Ensure that results are fetched from the queue after all processes are done
    while not q.empty():
        idx, result = q.get()
        results[idx] = result  # Store result in the correct order
    finalErrors = np.array(results)
    minEig = np.where(finalErrors == finalErrors.min())[0][-1]
    minEig = tempRange[minEig]
    return minEig, finalErrors



def read_network(file_path):
    return nx.read_edgelist(file_path)
    
def load_subgraph(file_path, main_map, rev_map, main_node_count):
    """
    加载子图并确保节点索引与主图一致
    
    :param file_path: 子图边列表文件路径
    :param main_map: 主图的节点映射 dict(原始节点 -> 整数索引)
    :param rev_map: 主图的反向映射 dict(整数索引 -> 原始节点)
    :param main_node_count: 主图的节点总数
    :return: 节点索引与主图一致的子图
    """
    try:
        # 1. 加载子图边列表 - 正确处理带权重的图
        subgraph = nx.read_edgelist(
            file_path, 
            nodetype=str, 
            data=[('weight', float)],  # 明确读取权重列
            create_using=nx.DiGraph()    # 确保创建正确的图类型
        )
        mapping = {}
        nodes_to_remove = []
        
        # 2. 创建子图节点到主图索引的映射
        for node in list(subgraph.nodes()):
            # 先尝试直接匹配原始节点
            if node in main_map:
                mapping[node] = main_map[node]
            else:
                nodes_to_remove.append(node)
        subgraph.remove_nodes_from(nodes_to_remove)
        
        # 4. 重新标记节点为主图索引
        nx.relabel_nodes(subgraph, mapping, copy=False)
        
        # 5. 确保包含所有主图节点(添加孤立节点)
        all_main_nodes = set(range(main_node_count))
        existing_nodes = set(subgraph.nodes())
        missing_nodes = all_main_nodes - existing_nodes
        subgraph.add_nodes_from(missing_nodes)
        
        return subgraph
    
    except Exception as e:
        # 如果读取失败，尝试直接加载权重 - 手动处理文件
        import re
        subgraph = nx.DiGraph()
        with open(file_path, 'r') as f:
            for line in f:
                parts = re.split(r'\s+', line.strip())
                if len(parts) < 2:
                    continue
                
                u = parts[0]
                v = parts[1]
                weight = float(parts[2]) if len(parts) >= 3 else 1.0
                
                # 只添加两端节点都在主图中的边
                if u in main_map and v in main_map:
                    u_index = main_map[u]
                    v_index = main_map[v]
                    subgraph.add_edge(u_index, v_index, weight=weight)
        
        # 确保包含所有主图节点(添加孤立节点)
        all_main_nodes = set(range(main_node_count))
        existing_nodes = set(subgraph.nodes())
        missing_nodes = all_main_nodes - existing_nodes
        subgraph.add_nodes_from(missing_nodes)
        
        return subgraph


def calculate_domirank(network, num_runs=1, directed=True):
    for _ in range(num_runs):
        GAdj = nx.to_scipy_sparse_array(network,weight = "weight")
        if directed:
            GAdj = sp.sparse.csr_array(GAdj.T)
        
        lambN = find_eigenvalue(GAdj, maxIter=1000, dt=0.01, checkStep=25)
        
        sigma, _ = optimal_sigma(GAdj, analytical=False, endVal=lambN)
        print("这里是正确的")
        domirank_dist,_ = domirank(GAdj, analytical=False, sigma=sigma)
    
    return domirank_dist, sigma