########## Here are the associated DomiRank functions #############
import inspect
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


def reduce_graph_by_weighted_equivalence(G: nx.DiGraph) -> nx.DiGraph:
    import networkx as nx
    from collections import defaultdict
    print("开始识别等价节点...")
    is_directed = G.is_directed()

    # 结构签名 -> [节点列表]
    signatures = defaultdict(list)

    for node in G.nodes():
        # --- 生成唯一的签名（兼容无向/有向图）---
        if is_directed:
            # 有向图：出邻居和入邻居均参与签名
            out_neigh = tuple(sorted((v, data.get('weight', 1.0)) 
                                    for u, v, data in G.out_edges(node, data=True)))
            in_neigh = tuple(sorted((u, data.get('weight', 1.0)) 
                                   for u, v, data in G.in_edges(node, data=True)))
            signature = (out_neigh, in_neigh)
        else:
            # 无向图：邻居和权重直接排序
            signature = tuple(sorted(
                (neighbor, data.get('weight', 1.0)) 
                for _, neighbor, data in G.edges(node, data=True)
            ))
        signatures[signature].append(node)

    # 生成节点映射（原始节点 -> 代表节点）
    node_to_representative = {
        node: min(nodes)  # 选择组内最小节点作为代表
        for nodes in signatures.values()
        for node in nodes
    }

    num_reduced_nodes = len(signatures)
    print(f"识别完成。节点数从 {G.number_of_nodes()} 减少到 {num_reduced_nodes}。")

    # --- 构建缩减图 ---
    reduced_G = G.__class__()  # 保持原图类型（无向/有向）
    reduced_G.add_nodes_from(set(node_to_representative.values()))  # 添加代表节点

    # 合并边权重
    edge_weights = defaultdict(float)
    for u, v, data in G.edges(data=True):
        ru = node_to_representative[u]
        rv = node_to_representative[v]
        weight = data.get('weight', 1.0)

        # 无向图需避免重复（例如 (a,b) 和 (b,a) 是同一条边）
        if not is_directed and ru > rv:
            ru, rv = rv, ru  # 统一排序

        edge_weights[(ru, rv)] += weight

    # 添加边到缩减图
    for (u, v), weight in edge_weights.items():
        reduced_G.add_edge(u, v, weight=weight)

    print("缩减图构建完成。")
    return reduced_G


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



def get_lcc_metrics(graph_or_matrix):
    
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    # --- 路径 A: 输入是 NetworkX Graph ---
    if isinstance(graph_or_matrix, nx.Graph):
        G = graph_or_matrix

        if G.number_of_nodes() == 0:
            return 0, 0, 0.0

        if G.number_of_edges() == 0:
            return 1, 0, 0.0

        # 对于有边的图，正常计算
        try:
            lcc_nodes = max(nx.connected_components(G), key=len)
            lcc_subgraph = G.subgraph(lcc_nodes)
            
            node_count = lcc_subgraph.number_of_nodes()
            link_count = lcc_subgraph.number_of_edges()
            weighted_flow = lcc_subgraph.size(weight='weight')
            return node_count, link_count, weighted_flow
        except ValueError: # 兜底，理论上不会触发了
            return 0, 0, 0.0

    # --- 路径 B: 输入是 SciPy 稀疏矩阵 ---
    elif isinstance(graph_or_matrix, (sp.csr_matrix, sp.csr_array)):
        GAdj = graph_or_matrix
        num_nodes = GAdj.shape[0]
        if num_nodes == 0:
            return 0, 0, 0.0
        if GAdj.nnz == 0:
            # LCC 是一个单独的节点
            return 1, 0, 0.0
            
        n_components, labels = sp.csgraph.connected_components(GAdj, directed=False)
        
        if n_components == 0:
            return 0, 0, 0.0

        # 【修正3】: 更安全地找到 LCC
        if labels.size == 0: # 再次检查，以防万一
             return 1, 0, 0.0 # 如果有节点但没有标签，LCC大小为1
             
        component_sizes = np.bincount(labels)
        lcc_label = np.argmax(component_sizes)
        lcc_node_count = component_sizes[lcc_label]
        
        # 如果最大的组件大小为0（理论上不会发生，但作为防御性编程）
        if lcc_node_count == 0 and num_nodes > 0:
            return 1, 0, 0.0

        lcc_indices = np.where(labels == lcc_label)[0]
        lcc_submatrix = GAdj[lcc_indices, :][:, lcc_indices]
        
        # 对于无向图的邻接矩阵，边数和总权重都需要除以2
        link_count = lcc_submatrix.nnz / 2
        weighted_flow = lcc_submatrix.sum() / 2
        return int(lcc_node_count), int(link_count), weighted_flow

    else:
        raise TypeError("输入必须是 NetworkX Graph 或 SciPy 稀疏矩阵")
        
def get_link_size(G):
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        links = len(G.edges()) #convert to scipy sparse if it is a graph 
    elif type(G) == scipy.sparse.csr_array:
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links


def remove_node(graph_or_matrix, nodes_to_remove):
    """
    【修正版】从 NetworkX 图或 SciPy 稀疏矩阵中移除一个节点列表。
    """
    
    import networkx as nx
    import scipy.sparse as sp
    # 确保 nodes_to_remove 是一个列表或集合，以便统一处理
    cleaned_nodes = [item for item in nodes_to_remove if isinstance(item, (int, np.integer))]

    if not cleaned_nodes: # 如果清洗后没有有效的节点，则直接返回
        return graph_or_matrix
    if not isinstance(nodes_to_remove, (list, set, tuple)):
        nodes_to_remove = [nodes_to_remove]

    if isinstance(graph_or_matrix, nx.Graph):
        # NetworkX 的 remove_nodes_from 可以高效地处理列表
        graph_or_matrix.remove_nodes_from(nodes_to_remove)
        return graph_or_matrix
        
    elif isinstance(graph_or_matrix, (sp.csr_matrix, sp.csr_array)):
        valid_nodes = [node for node in cleaned_nodes if 0 <= node < graph_or_matrix.shape[0]]
        if not valid_nodes:
            return graph_or_matrix

        mask = np.ones(graph_or_matrix.shape[0], dtype=bool)
        mask[valid_nodes] = False
        
        diag_matrix = sp.diags(mask.astype(int), offsets=0, format='csr')
        
        graph_or_matrix = diag_matrix @ graph_or_matrix @ diag_matrix
        return graph_or_matrix
        
    else:
        raise TypeError("输入必须是 NetworkX Graph 或 SciPy CSR Matrix/Array")


def generate_attack_1(scores, node_list):
    node_score_pairs = list(zip(node_list, scores))
    sorted_pairs = sorted(node_score_pairs, key=lambda item: item[1], reverse=True)
    return [node for node, score in sorted_pairs]



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

def simulate_attack(graph, attack_order):
    """
    根据给定的节点移除顺序，模拟攻击并记录最大连通子图的变化。
    """
    g_copy = graph.copy()
    original_lcc_size = len(max(nx.connected_components(g_copy), key=len))
    lcc_sizes = [original_lcc_size]
    
    # 确保我们模拟的步数与攻击列表一致
    for node_to_remove in attack_order:
        if g_copy.has_node(node_to_remove):
            g_copy.remove_node(node_to_remove)
            if g_copy.number_of_nodes() > 0:
                current_lcc_size = len(max(nx.connected_components(g_copy), key=len))
                lcc_sizes.append(current_lcc_size)
            else:
                lcc_sizes.append(0)
    
    # 补齐因节点不存在而可能缺失的步骤
    while len(lcc_sizes) < len(attack_order) + 1:
        lcc_sizes.append(lcc_sizes[-1])

    return [size / original_lcc_size for size in lcc_sizes]

def network_attack_sampled(GAdj, attackStrategy, sampling=0, target_lcc_ratio=0.5, target_flow_ratio=0.5):
    """
    【最终修正版】网络攻击模拟，采用正确的增量攻击逻辑。
    """
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    is_graph = isinstance(GAdj, nx.Graph)
    N = GAdj.number_of_nodes() if is_graph else GAdj.shape[0]
    initialComponent, initialLinks, initialWeightedFlow = get_lcc_metrics(GAdj)

    if sampling == 0:
        sampling = 1 if N < 100 else max(1, int(N / 100))

    # --- 1. 在循环外只复制一次图 ---
    GAdj_copy = GAdj.copy()

    # --- 2. 初始化结果存储 ---
    componentEvolution = [initialComponent]
    linksEvolution = [initialLinks]
    weightedFlowEvolution = [initialWeightedFlow]
    removed_ratios = [0.0]
    lcc_critical_ratio = None
    flow_critical_ratio = None
    
    attack_strategy_q = list(attackStrategy)
    
    while attack_strategy_q:
        batch_to_remove = attack_strategy_q[:sampling]
        # 更新攻击列表，移除已经取出的节点
        attack_strategy_q = attack_strategy_q[sampling:]
        GAdj_copy = remove_node(GAdj_copy, batch_to_remove) # 假设remove_node能处理列表
        
        # 3c. 在循环的每次迭代中评估并记录一次
        num_removed = N - len(attack_strategy_q)
        comp_size, link_size, weighted_flow = get_lcc_metrics(GAdj_copy)
            
        comp_ratio = comp_size #/ initialComponent if initialComponent > 0 else 0
        link_ratio = link_size #/ initialLinks if initialLinks > 0 else 0
        weighted_flow_ratio = weighted_flow#/ initialWeightedFlow if initialWeightedFlow > 0 else 0
        removal_ratio = num_removed / N
        
        componentEvolution.append(comp_ratio)
        linksEvolution.append(link_ratio)
        weightedFlowEvolution.append(weighted_flow_ratio)
        removed_ratios.append(removal_ratio)
    
        # 3d. 检查临界点和中断条件
        if lcc_critical_ratio is None and comp_ratio <= target_lcc_ratio:
            lcc_critical_ratio = removal_ratio
        if flow_critical_ratio is None and weighted_flow_ratio <= target_flow_ratio:
            flow_critical_ratio = removal_ratio
        
        if comp_size <= 1:
            break
    
    componentEvolution = [size / initialComponent for size in componentEvolution]
    linksEvolution = [size / initialLinks for size in linksEvolution]
    weightedFlowEvolution = [size / initialWeightedFlow for size in weightedFlowEvolution]
    # --- 4. 返回字典 ---
    return {
        "lcc_critical_ratio": lcc_critical_ratio,
        "flow_critical_ratio": flow_critical_ratio,
        "removed_ratios": removed_ratios,
        "components": np.array(componentEvolution),
        "links": np.array(linksEvolution),
        "weighted_flow": np.array(weightedFlowEvolution),
        "attack_sequence": attackStrategy # 返回原始的完整攻击序列
    }





#############################################################################方案一，解析解优化，利用贪心策略###############################################



# 确保导入所有需要的库
from scipy.sparse.linalg import eigs, gmres, cg
import inspect
from collections import deque
def sfs_pruning(G, freeze_quantile=None, max_weight_threshold=1):
    import numpy as np
    import scipy.sparse as sp
    N = G.shape[0]
    if N == 0:
        return np.array([], dtype=bool)
    G_csr = G.tocsr() if not sp.isspmatrix_csr(G) else G
    if freeze_quantile is None:
        G_adj = (G_csr + G_csr.T).astype(bool)
        G_adj.setdiag(False)
        G_adj_csr = G_adj.tocsr()
        G_adj_csr.eliminate_zeros()
        current_degrees = np.array(G_adj_csr.indptr[1:] - G_adj_csr.indptr[:-1])
        active_mask = np.ones(N, dtype=bool)
        peel_queue = deque(np.where(current_degrees <= 1)[0])
        active_mask[peel_queue] = False
        num_frozen_by_core = len(peel_queue)
        while peel_queue:
            v = peel_queue.popleft()
            for u in G_adj_csr.indices[G_adj_csr.indptr[v]:G_adj_csr.indptr[v+1]]:
                if active_mask[u]:
                    current_degrees[u] -= 1
                    if current_degrees[u] == 1:
                        active_mask[u] = False
                        peel_queue.append(u)
                        num_frozen_by_core += 1
        active_mask_after_core_pruning = active_mask.copy()
        out_degrees = G_csr.indptr[1:] - G_csr.indptr[:-1]
        max_weights_per_node = np.zeros(N)
        has_edges_mask = out_degrees > 0
        if np.any(has_edges_mask):
            max_weights_subset = np.maximum.reduceat(G_csr.data, G_csr.indptr[:-1][has_edges_mask])
            max_weights_per_node[has_edges_mask] = max_weights_subset
        mask_to_freeze = active_mask & (max_weights_per_node < max_weight_threshold)
        indices_to_freeze_by_weight = np.where(mask_to_freeze)[0]
        if len(indices_to_freeze_by_weight) > 0:
            active_mask[indices_to_freeze_by_weight] = False
        num_frozen_total = N - np.sum(active_mask)
        if num_frozen_total > 0.8 * N:
            active_mask = active_mask_after_core_pruning
    elif 0 <= freeze_quantile <= 1:
        node_strengths = np.array(G_csr.sum(axis=1)).flatten()
        num_to_freeze = int(N * freeze_quantile)
        if num_to_freeze == 0: return np.ones(N, dtype=bool)
        if num_to_freeze >= N: return np.zeros(N, dtype=bool)
        indices_to_freeze = np.argpartition(node_strengths, num_to_freeze - 1)[:num_to_freeze]
        active_mask = np.ones(N, dtype=bool)
        active_mask[indices_to_freeze] = False
    else:
        raise ValueError("freeze_quantile 必须是 [0, 1] 范围内的浮点数或 None。")
    return active_mask




# 辅助函数，用于检查矩阵对称性
def is_symmetric(A, tol=1e-12):
    """
    检查一个（可能稀疏的）矩阵是否对称。
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs, gmres, cg
    if not sp.issparse(A):
        return np.allclose(A, A.T, atol=tol)
    if not isinstance(A, (sp.csr_matrix, sp.csc_matrix)):
        A = A.tocsr() # 转换为CSR格式以便进行高效操作
    if A.shape[0] != A.shape[1]: 
        return False
    diff = A - A.T
    return diff.nnz == 0 or np.all(np.abs(getattr(diff, 'data', [])) < tol)


import numpy as np
import scipy.sparse as sp
# 确保在您的环境中可以导入 gmres
from scipy.sparse.linalg import gmres 


def adaptive_greedy_domirank_refined(G, sigma, precomputed_rho_A=None, 
                                     max_K=80, greedy_epsilon=1e-12, 
                                     correction_solver_iter=100,
                                     active_mask=None):
    """
    使用 Krylov 子空间方法直接求解 DomiRank。
    此版本在图对称时，使用一个自定义的共轭梯度法实现，
    以清晰地展示和控制精度。
    """
    import scipy.sparse as sp

    # =========================================================================
    # == 自实现的共轭梯度 (CG) 求解器 ==
    # =========================================================================
    def my_conjugate_gradient(A, b, tol=1e-12, maxiter=None):
        """
        一个清晰、自包含的共轭梯度法实现，用于求解 Ax = b。
        A 必须是稀疏、对称且正定的矩阵。
        """
        N = A.shape[0]
        
        # 1. 初始化
        x = np.zeros(N, dtype=np.float64)  # 初始解向量 x_0 = 0
        r = b - A.dot(x)                  # 初始残差 r_0 = b - Ax_0 = b
        p = r.copy()                      # 初始搜索方向 p_0 = r_0
        rs_old = np.dot(r, r)             # 残差的平方和 (r_0^T * r_0)

        # 如果初始残差已经很小，说明 b 本身就是0向量，直接返回
        if rs_old < 1e-12:
            return x, 0

        # --- 这是我们的精度度量标准 ---
        # 收敛条件是： ||r_k|| <= tol * ||b||
        # 我们计算 b 的范数（长度）用于后续的相对误差比较
        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            b_norm = 1.0 # 避免除以零
        
        # 确定最大迭代次数
        effective_maxiter = maxiter if maxiter is not None and maxiter > 0 else N * 10

        # 2. 核心迭代循环
        for i in range(effective_maxiter):
            Ap = A.dot(p)
            
            # 计算步长 alpha
            alpha = rs_old / np.dot(p, Ap)
            
            # 更新解和残差
            x += alpha * p
            r -= alpha * Ap
            
            rs_new = np.dot(r, r)
            residual_norm = np.sqrt(rs_new)
            if residual_norm < tol * b_norm:
                print(f"自定义 CG 求解器在第 {i+1} 次迭代成功收敛。")
                return x, 0
            p = r + (rs_new / rs_old) * p
            
            rs_old = rs_new
        print(f"警告：自定义 CG 达到最大迭代次数 {effective_maxiter}，但未收敛。")
        return x, i + 1 # exitCode > 0 表示未收敛
    original_N = G.shape[0]
    if original_N == 0:
        return True, np.array([])

    if active_mask is not None and np.any(active_mask):
        active_count = np.sum(active_mask)
        if active_count == 0:
            return True, np.zeros(original_N)
        print(f"检测到 active_mask，将在 {active_count} 个核心节点的子图上进行计算。")
        
        original_indices_map = np.where(active_mask)[0]
        G_work = G[active_mask][:, active_mask]
        is_pruned_mode = True
    else:
        G_work = G
        is_pruned_mode = False

    N = G_work.shape[0]
    B = sigma * G_work + sp.identity(N, format='csr')
    v = sigma * np.asarray(G_work.sum(axis=1)).flatten()

    max_iterations = correction_solver_iter if correction_solver_iter > 0 else None
    Psi_solved = None
    exitCode = -1

    try:
        if is_symmetric(G_work) : #!= 1 :

            print("图是对称的，使用自定义的共轭梯度法 (my_conjugate_gradient) 求解...")
            Psi_solved, exitCode = my_conjugate_gradient(B, v, 
                                                       tol=greedy_epsilon, 
                                                       maxiter=max_iterations)
        else:
            print(f"图非对称，使用 SciPy 的通用最小残差法 (GMRES) 求解...")
            # 这里的兼容性代码保持不变
            try:
                Psi_solved, exitCode = gmres(B, v, tol=greedy_epsilon, maxiter=max_iterations)
            except TypeError:
                Psi_solved, exitCode = gmres(B, v, maxiter=max_iterations)

    except Exception as e:
        print(f"在 Krylov 求解过程中发生严重异常: {e}")
        return False, np.zeros(original_N)
        
    # --- 后续处理逻辑完全保持不变 ---
    if exitCode == 0:
        # 自定义求解器成功时也会打印信息，这里可以保留
        print("Krylov 求解器成功收敛。")
    elif exitCode > 0:
        print(f"警告：求解器达到最大迭代次数 {exitCode}，但未收敛到期望的容忍度。结果可能不准确。")
    else:
        print(f"错误：Krylov 求解器计算失败，退出码: {exitCode}。可能是由于数值问题。")
        return False, np.zeros(original_N)

    if is_pruned_mode:
        final_scores = np.zeros(original_N)
        final_scores[original_indices_map] = np.asarray(Psi_solved).flatten()
    else:
        final_scores = Psi_solved
        
    return True, final_scores



def domirank(G,analytical = False, sigma = -1, dt=0.1, epsilon=1e-12, maxIter = 1000, checkStep = 10,lambN = None,active_mask=None ):
    import numpy as np
    import networkx as nx
    import scipy.sparse as sp
    if not sp.issparse(G):
        if isinstance(G, nx.classes.graph.Graph):
            G_adj = nx.to_scipy_sparse_array(G, weight="weight")
        else:
            raise TypeError("输入 G 必须是 networkx Graph 或 scipy 稀疏矩阵")
    else:
        G_adj = G.copy()
    N = G_adj.shape[0]
    if analytical:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        _,Psi = adaptive_greedy_domirank_refined(G, sigma, precomputed_rho_A=lambN, 
                                     max_K=2, greedy_epsilon=epsilon, active_mask=active_mask,
                                     correction_solver_iter=180)
        return True, Psi
    else:
        if active_mask is None:
            active_mask = sfs_pruning(G_adj)
        else:
            active_mask = active_mask
        active_indices = np.where(active_mask)[0]
        frozen_indices = np.where(~active_mask)[0]
        num_active = len(active_indices)
        Psi = np.zeros(N, dtype=np.float64)
        if num_active == 0: return True, np.zeros(N)
        if num_active == N: print("  -> 所有节点都为活跃状态，执行标准全图计算。")
        if sigma == -1: sigma, _ = optimal_sigma(G_adj, analytical=False)
        pGAdj = (sigma * G_adj).astype(np.float64)
        pGAdj_active = pGAdj[active_indices, :][:, active_indices]
        row_sums_active = np.asarray(pGAdj_active.sum(axis=1)).flatten()
        if len(frozen_indices) > 0:
            G_af = G_adj[active_indices, :][:, frozen_indices]
            pGAdj_af = (sigma * G_af).astype(np.float64)
            Psi_frozen_initial = Psi[frozen_indices]
            influence_from_frozen = pGAdj_af @ (1.0 - Psi_frozen_initial)
        else:
            influence_from_frozen = 0.0
        Psi_active = np.ones(num_active, dtype=np.float64) / num_active
        dt = np.float64(dt)
        boundary = epsilon * num_active
        j= 0
        maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)

        for i in range(maxIter):
            influence_active = (row_sums_active + influence_from_frozen) - (pGAdj_active @ Psi_active)
            tempVal_active = (influence_active - Psi_active) * dt
            Psi_active += tempVal_active
            if i % checkStep == 0 and i != 0:
                if np.abs(tempVal_active).sum() < boundary:
                    print(f"在第 {i} 次迭代收敛。")
                    break

                if j < len(maxVals):
                    maxVals[j] = np.abs(tempVal_active).max()
                    if j > 1 and maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2] and maxVals[j-2] > maxVals[j-3]:
                        print("发散")
                        return False, Psi_active
                    j += 1

                max_change = np.abs(tempVal_active).max()
                if not np.isfinite(max_change):
                    print(f"错误：在第 {i} 次迭代时计算结果发散 (NaN/inf)。")
                    return False, Psi_active
        Psi[active_indices] = Psi_active
        if len(frozen_indices) > 0:
            full_row_sums = np.asarray(pGAdj.sum(axis=1)).flatten()
            influence_on_frozen = full_row_sums[frozen_indices] - (pGAdj[frozen_indices, :] @ Psi)
            Psi[frozen_indices] += (influence_on_frozen - Psi[frozen_indices])   
        return True, Psi

    
import scipy as sp 
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 60, dt = 0.1, epsilon = 1e-6, maxIter = 800, checkStep = 10,active_mask=None):
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
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep,active_mask)[0]:
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
import multiprocessing as mp
import numpy as np

# --- 1. 定义全局变量 ---
# 我们将在这里存放共享的大矩阵
shared_spArray = None

def init_worker(spArray_from_main):
    global shared_spArray
    shared_spArray = spArray_from_main

def process_iteration_shared(sigma, analytical, maxIter, checkStep, dt, epsilon, sampling,lambN,active_mask):
    """
    【内存安全版】的并行任务函数。
    它不再接收 spArray 作为参数，而是直接从全局变量中读取。
    """
    global shared_spArray # 声明我们要使用全局的矩阵
    
    try:
        tf, domiDist = domirank(shared_spArray, analytical=analytical, sigma=sigma, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep,lambN = lambN,active_mask=active_mask)
        if tf:
            domiAttack = generate_attack(domiDist)
            ourTempAttack = network_attack_sampled(shared_spArray, domiAttack, sampling=sampling)
            num_to_sum = int(len(ourTempAttack['components']) * 1.0)
            finalError = np.array(ourTempAttack['components'])[:num_to_sum].sum()
            print(f"Sigma={sigma:.6f} 计算完成, AUC={finalError:.4f}")
        else:
            finalError = 450.0
            print(f"Sigma={sigma:.6f} 未收敛。")
        return (sigma, finalError, tf) # tf 是收敛标志
        
    except Exception as e:
        print(f"Sigma={sigma:.6f} 的计算失败: {e}")
        return None

def optimal_sigma(spArray, analytical=True, endVal=0, startval=0.000001, iterationNo=80, dt=1e-4, epsilon=1e-6, maxIter=800, checkStep=10, maxDepth=100, sampling=0,lambN=None,active_mask=None ):
    """
    【内存安全版】的最优 sigma 搜索函数。
    """
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth=maxDepth, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep,lambN = lambN,active_mask=active_mask)

    endval = -0.9999 / endVal
    tempRange = np.linspace(startval, endval, iterationNo)
    
    max_processes = 5
    print(f"启动进程池，使用 {max_processes} 个工作进程...")

    # --- 关键改动在这里 ---
    # 1. 准备传递给子进程的、不包含大矩阵的参数
    tasks = [(sigma, analytical, maxIter, checkStep, dt, epsilon, sampling,lambN,active_mask) for sigma in tempRange]
    with mp.Pool(processes=max_processes, 
                 initializer=init_worker, 
                 initargs=(spArray,)) as pool:
        
        # 3. starmap 现在调用的是内存安全版的函数，并且只传递小参数
        results = pool.starmap(process_iteration_shared, tasks)
    
    # --- 结果处理 (使用之前建议的健壮版本) ---
    converged_results = []
    unconverged_results = []
    for res in results:
        if res is not None:
            sigma, auc_error, is_converged = res
            if is_converged:
                converged_results.append((sigma, auc_error))
            else:
                unconverged_results.append((sigma, auc_error))

    if not converged_results:
        raise RuntimeError("所有任务都未能成功收敛！请检查 dt 和 maxIter 参数。")

    best_sigma, min_error = min(converged_results, key=lambda item: item[1])
    
    print(f"\n计算完成。在【成功收敛】的结果中，找到的最优Sigma为: {best_sigma:.6f}, 对应的最小误差(AUC)为: {min_error:.4f}")

    all_errors = {sigma: error for sigma, error, conv in results if res is not None}
    finalErrors_full = np.array([all_errors.get(s, np.inf) for s in tempRange])

    return best_sigma, finalErrors_full