########## Here are the associated DomiRank functions #############
import gc
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

# def remove_node(G, removedNode):
#     if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
#         if G.has_node(removedNode):
#             G.remove_node(removedNode)
#         else:
#             for node in removedNode:
#                 G.remove_node(node) #remove node in graph form
#         return G
#     elif type(G) == scipy.sparse.csr_array:
#         diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0])) 
#         diag[removedNode, removedNode] = 0 #set the rows and columns that are equal to zero in the sparse array
#         G = diag @ G 
#         return G @ diag


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


def sfs_pruning(G, freeze_quantile=None, max_weight_threshold=1):
    import numpy as np
    import scipy.sparse as sp
    from collections import deque
    """
    自动模式逻辑:
    1. 拓扑剪枝: (高效实现) 冻结所有核数为1的节点。
    2. 最大权重剪枝: 在剩余的核心节点中，冻结最强出边权重低于阈值的节点。
    3. 安全检查: 若总冻结比例过高，则回退到仅使用拓扑剪枝的结果。
    """
    N = G.shape[0]
    if N == 0:
        return np.array([], dtype=bool)

    # 确保是CSR格式，以实现最高效的行操作
    G_csr = G.tocsr() if not sp.isspmatrix_csr(G) else G

    # --- 自动模式 ---
    if freeze_quantile is None:
        print("--- 自动模式 (混合策略: 核数 + 最大权重阈值) ---")
        
        # --- 阶段1: 高效拓扑剪枝 (迭代剥离叶子节点，无NetworkX) ---
        print("  - 阶段1/2: 正在基于'剥离叶子节点'算法冻结最外围节点...")
        
        # 1. 创建一个无向、无自环的邻接矩阵用于度数计算
        #    (G_csr + G_csr.T) > 0 会创建一个对称的、二值化的布尔稀疏矩阵
        G_adj = (G_csr + G_csr.T).astype(bool)
        G_adj.setdiag(False)
        G_adj_csr = G_adj.tocsr()
        G_adj_csr.eliminate_zeros()

        # 2. 计算每个节点的初始度数 (这是一个可修改的Numpy数组)
        current_degrees = np.array(G_adj_csr.indptr[1:] - G_adj_csr.indptr[:-1])

        # 3. 默认所有节点都是活跃的 (core > 1)
        active_mask = np.ones(N, dtype=bool)
        
        # 4. 初始化剥离队列：将所有度数 <= 1 的节点加入队列
        peel_queue = deque(np.where(current_degrees <= 1)[0])
        
        # 5. 将这些初始节点标记为不活跃 (冻结)
        active_mask[peel_queue] = False
        num_frozen_by_core = len(peel_queue)

        # 6. 开始迭代剥离过程，直到队列为空
        while peel_queue:
            v = peel_queue.popleft()  # 弹出一个已冻结的节点 v

            # 遍历 v 的所有邻居 u
            for u in G_adj_csr.indices[G_adj_csr.indptr[v]:G_adj_csr.indptr[v+1]]:
                if active_mask[u]:
                    # 如果邻居 u 仍活跃，其度数减 1
                    current_degrees[u] -= 1
                    # 如果 u 的度数因此降为1，它就成了新的叶子节点
                    if current_degrees[u] == 1:
                        active_mask[u] = False # 冻结 u
                        peel_queue.append(u)   # 将 u 加入队列
                        num_frozen_by_core += 1

        print(f"    -> 通过'剥离叶子节点'算法，冻结了 {num_frozen_by_core} 个节点。")
        
        # **重要**: 保存仅经过核心剪枝的结果，以备安全回退时使用
        active_mask_after_core_pruning = active_mask.copy()

        # --- 阶段2: 基于最大边权重的二次剪枝 (这部分已经很高效，无需改动) ---
        print(f"  - 阶段2/2: 在剩余活跃节点中，基于'最大边权重' (阈值 < {max_weight_threshold}) 进行二次剪枝...")
        out_degrees = G_csr.indptr[1:] - G_csr.indptr[:-1]
        max_weights_per_node = np.zeros(N)
        has_edges_mask = out_degrees > 0
        
        if np.any(has_edges_mask):
            max_weights_subset = np.maximum.reduceat(G_csr.data, G_csr.indptr[:-1][has_edges_mask])
            max_weights_per_node[has_edges_mask] = max_weights_subset
            
        mask_to_freeze = active_mask & (max_weights_per_node < max_weight_threshold)
        indices_to_freeze_by_weight = np.where(mask_to_freeze)[0]

        if len(indices_to_freeze_by_weight) > 0:
            print(f"    -> 通过最大权重剪枝，额外冻结了 {len(indices_to_freeze_by_weight)} 个节点。")
            active_mask[indices_to_freeze_by_weight] = False
        else:
            print("    -> 没有节点的'最大权重'低于阈值，无需额外冻结。")

        # --- 阶段3: 安全限制 (无需改动) ---
        num_frozen_total = N - np.sum(active_mask)
        if num_frozen_total > 0.8 * N:
            print(f"警告：总冻结比例 ({num_frozen_total/N*100:.1f}%) 过高。")
            print(f"       将放弃阶段2的权重剪枝，回退到仅使用核数剪枝的安全结果。")
            active_mask = active_mask_after_core_pruning

    # --- 手动模式 (已经最优，无需改动) ---
    elif 0 <= freeze_quantile <= 1:
        print(f"--- 手动模式 (基于强度): 正在冻结强度最低的 {freeze_quantile*100:.1f}% 的节点 ---")
        node_strengths = np.array(G_csr.sum(axis=1)).flatten()
        num_to_freeze = int(N * freeze_quantile)
        if num_to_freeze == 0: return np.ones(N, dtype=bool)
        if num_to_freeze >= N: return np.zeros(N, dtype=bool)
        indices_to_freeze = np.argpartition(node_strengths, num_to_freeze - 1)[:num_to_freeze]
        active_mask = np.ones(N, dtype=bool)
        active_mask[indices_to_freeze] = False
    else:
        raise ValueError("freeze_quantile 必须是 [0, 1] 范围内的浮点数或 None。")

    num_active = np.sum(active_mask)
    num_frozen = N - num_active
    print(f"剪枝完成 -> 活跃节点: {num_active} ({num_active/N*100:.2f}%), 冻结节点: {num_frozen} ({num_frozen/N*100:.2f}%)")
    return active_mask

#######################################################压缩图方法###################################################

import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict, deque

# --- 模块级缓存，用于存储预计算结果 ---
_COMPRESSION_CACHE = {}

def find_congruent_vertices_iterative(G: sp.spmatrix):
    """
    通过迭代划分细化算法寻找“输出全等”的节点集 (最大压缩率版)。
    该版本使用NumPy向量化操作加速了签名计算过程，并优化了内存使用。
    增加了针对“巨型块分裂”的性能优化。
    """
    print("--- 正在通过迭代划分细化算法寻找“输出全等”节点集 (NumPy加速与内存优化版)... ---")

    N = G.shape[0]
    if N == 0:
        return [], []
        
    G_csr = G.tocsr()
    is_directed = (G != G.T).nnz > 0

    # --- 1. 初始划分 ---
    out_degrees = G_csr.indptr[1:] - G_csr.indptr[:-1]
    partitions = defaultdict(list)
    for i in range(N):
        partitions[out_degrees[i]].append(i)
    blocks = list(partitions.values())
    node_to_block_id = np.empty(N, dtype=np.int32)
    for i, block in enumerate(blocks):
        node_to_block_id[block] = i
    print(f"初始划分完成，共有 {len(blocks)} 个块。")

    # --- 2. 迭代细化 ---
    iteration = 0
    # 定义一个阈值，大于这个尺寸的块将使用NumPy排序策略
    NUMPY_SPLIT_THRESHOLD = 500 

    while True:
        iteration += 1
        has_changed = False
        new_blocks = []
        
        print(f"--- 开始迭代 {iteration}, 当前块数: {len(blocks)}, 最大块尺寸: {max(len(b) for b in blocks if b)} ---")

        for block_idx, block in enumerate(blocks):
            if (block_idx + 1) % 100000 == 0:
                 print(f"  已处理 {block_idx + 1} / {len(blocks)} 个块...")

            if len(block) <= 1:
                if block: new_blocks.append(block)
                continue
            if len(block) > NUMPY_SPLIT_THRESHOLD:
                block_nodes = np.array(block, dtype=np.int32)
                # 使用64位无符号整数存储哈希值，更健壮
                signatures = np.empty(len(block), dtype=np.int64) 

                for i, node_idx in enumerate(block_nodes):
                    start, end = G_csr.indptr[node_idx], G_csr.indptr[node_idx+1]
                    out_indices = G_csr.indices[start:end]
                    if out_indices.size == 0:
                        signatures[i] = 0
                        continue
                    out_weights = G_csr.data[start:end]
                    target_block_ids = node_to_block_id[out_indices]
                    sorted_indices = np.argsort(target_block_ids)
                    sorted_blocks = target_block_ids[sorted_indices]
                    sorted_weights = out_weights[sorted_indices]
                    signature_bytes_tuple = (sorted_blocks.tobytes(), sorted_weights.tobytes())
                    signatures[i] = hash(signature_bytes_tuple)
                
                # 核心：排序以分组
                # 'kind='mergesort'' 保证了稳定性，虽然在这里不是必须，但是是好习惯
                sorted_order = np.argsort(signatures, kind='mergesort')
                sorted_signatures = signatures[sorted_order]
                sorted_nodes = block_nodes[sorted_order]
                
                split_points = np.where(np.diff(sorted_signatures) != 0)[0] + 1
                
                # 使用边界来切分节点数组
                sub_blocks_arrays = np.split(sorted_nodes, split_points)

                if len(sub_blocks_arrays) > 1:
                    has_changed = True

                for sub_array in sub_blocks_arrays:
                    new_blocks.append(sub_array.tolist()) # 转换回list

            # --- 原有的字典策略 (适用于小块) ---
            else:
                signatures_in_block = defaultdict(list)
                for node_idx in block:
                    start, end = G_csr.indptr[node_idx], G_csr.indptr[node_idx+1]
                    out_indices = G_csr.indices[start:end]
                    if out_indices.size == 0:
                        signature_hash = 0 
                        signatures_in_block[signature_hash].append(node_idx)
                        continue
                    out_weights = G_csr.data[start:end]
                    target_block_ids = node_to_block_id[out_indices]
                    sorted_indices = np.argsort(target_block_ids)
                    sorted_blocks = target_block_ids[sorted_indices]
                    sorted_weights = out_weights[sorted_indices]
                    signature_bytes_tuple = (sorted_blocks.tobytes(), sorted_weights.tobytes())
                    signature_hash = hash(signature_bytes_tuple)
                    signatures_in_block[signature_hash].append(node_idx)

                if len(signatures_in_block) > 1:
                    has_changed = True
                
                for sub_block in signatures_in_block.values():
                    new_blocks.append(sub_block)
        
        print(f"迭代 {iteration} 完成: 块数量从 {len(blocks)} -> {len(new_blocks)}")
        if not has_changed:
            print("分区稳定，迭代结束。")
            break
            
        blocks = new_blocks
        if not blocks:
            break
        for i, block in enumerate(blocks):
            node_to_block_id[block] = i
    print("构建最终结果 (使用高效内部边检查)...")
    congruent_sets = []
    unique_nodes = []
    for block in blocks:
        if len(block) > 1:
            has_internal_edge = False
            if not is_directed:
                block_set = set(block)
                for u in block:
                    start, end = G_csr.indptr[u], G_csr.indptr[u+1]
                    neighbors_of_u = G_csr.indices[start:end]
                    for v in neighbors_of_u:
                        if v in block_set:
                            has_internal_edge = True
                            break # 结束内层循环
                    
                    if has_internal_edge:
                        break
                if has_internal_edge:
                    unique_nodes.extend(block)
                    continue 
            congruent_sets.append(block)
        else:
            unique_nodes.extend(block)
            
    print(f"迭代收敛。共找到 {len(congruent_sets)} 个输出全等节点集和 {len(unique_nodes)} 个唯一节点。")
    return congruent_sets, unique_nodes

def precompute_compression_data(G: sp.spmatrix):
    global _COMPRESSION_CACHE
    import scipy.sparse as sp
    # graph_hash = (G.data.tobytes(), G.indices.tobytes(), G.indptr.tobytes())
    import hashlib
    hasher = hashlib.sha256()
    hasher.update(G.data.tobytes())
    hasher.update(G.indices.tobytes())
    hasher.update(G.indptr.tobytes())
    graph_hash = hasher.hexdigest()
    if graph_hash in _COMPRESSION_CACHE:
        return _COMPRESSION_CACHE[graph_hash]

    print("--- 首次执行非对称图压缩 (混合构建优化版) ---")

    congruent_sets, unique_nodes = find_congruent_vertices_iterative(G)
    N = G.shape[0]
    if not congruent_sets:
        print("  -> 未找到可压缩的节点集。将使用原始图进行计算。")
        plan = {
            'is_compressible': False,
            'G_comp': G,  # G_comp 就是原始图 G
            'unique_nodes': np.arange(N), # 所有节点都是唯一的
            'congruent_sets': [], # 没有超节点
            'num_unique': N,
            'num_supernodes': 0
        }
        _COMPRESSION_CACHE[graph_hash] = plan
        return plan

    N = G.shape[0]
    num_unique = len(unique_nodes)
    num_supernodes = len(congruent_sets)
    N_comp = num_unique + num_supernodes
    representatives = [s[0] for s in congruent_sets]
    
    print(f"正在使用混合构建法构建非对称压缩图 (大小: {N_comp}x{N_comp})...")

    
    source_rows_indices = np.concatenate([unique_nodes, representatives])
    

    G_sub = G[source_rows_indices, :]
    node_to_new_col_idx = np.empty(N, dtype=np.int32)
    # unique_nodes_arr = np.array(unique_nodes)
    unique_nodes_arr = np.array(unique_nodes, dtype=int)
    node_to_new_col_idx[unique_nodes_arr] = np.arange(num_unique)
    for i, s in enumerate(congruent_sets):
        supernode_idx = num_unique + i
        node_to_new_col_idx[s] = supernode_idx
        
    print(f"将子图转换为COO格式，以方便地访问其列索引")

    G_sub_coo = G_sub.tocoo()
    del G_sub
    gc.collect()
    final_rows = G_sub_coo.row
    final_cols = node_to_new_col_idx[G_sub_coo.col]
    
    # --- 3. 一步构建：生成最终的压缩图 ---
    print(f"一步构建：生成最终的压缩图")
    G_comp = sp.csr_matrix((G_sub_coo.data, (final_rows, final_cols)), shape=(N_comp, N_comp))
    del G_sub_coo, final_cols # 这些中间数组也不再需要了
    gc.collect()
    G_comp.eliminate_zeros()
    
    # 打包 plan
    plan = {
        'is_compressible': True,
        'G_comp': G_comp,
        'unique_nodes': unique_nodes_arr, 
        'congruent_sets': congruent_sets,
        'num_unique': num_unique,
        'num_supernodes': num_supernodes
    }
    
    _COMPRESSION_CACHE[graph_hash] = plan
    return plan


def anderson_accel(g_evals, x_evals, m):
    """
    根据过去 m 步的历史，计算安德森加速后的下一个点。
    code
    Code
    参数:
    g_evals (list): 包含过去 m 个 g(x) 评估结果的列表 [g(x_k-m+1), ..., g(x_k)]
    x_evals (list): 包含过去 m 个状态 x 的列表 [x_k-m+1, ..., x_k]
    m (int): 使用的历史步数，即 len(g_evals)

    返回:
    np.array: 加速后的下一个点 x_{k+1}
    """
    # 1. 构建残差历史矩阵 F
    # F 的第 i 列是 g(x_i) - x_i
    F = np.array([g - x for g, x in zip(g_evals, x_evals)]).T
    delta_F = F[:, 1:] - F[:, :-1]

    # 构造最新的残差 f_k
    f_k = F[:, -1]
    try:
        gamma, _, rank, _ = np.linalg.lstsq(delta_F, -f_k, rcond=1e-8)
        if rank < delta_F.shape[1]:
             print(f"    [Warning] Anderson acceleration failed: History is rank-deficient ({rank} < {delta_F.shape[1]}). Skipping acceleration.")
             return None

    except np.linalg.LinAlgError:
        print("    [Warning] Anderson acceleration failed: Linear algebra error. Skipping acceleration.")
        return None
    g_k = g_evals[-1]
    prev_g_evals = np.array(g_evals[:-1]).T
    delta_G = prev_g_evals - g_k[:, np.newaxis] 
    next_x = g_k + delta_G @ gamma

    return next_x





def domirank(G, analytical=False, sigma=-1, dt=0.1, epsilon=1e-6, maxIter=800, checkStep=10, 
                               freeze_quantile=None,compression_plan = None,precomputed_mask=None,omega = None,
             use_anderson=True, anderson_freq=10, anderson_memory=5): 
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    import time
    
    is_graph_input = False
    if sp.issparse(G):
        G_adj = G.copy()
    elif isinstance(G, nx.classes.graph.Graph):
        node_list = list(G.nodes())
        is_graph_input = True
        G_adj = nx.to_scipy_sparse_array(G, weight="weight")
    else:
        raise TypeError("不支持的输入类型。")
    N = G_adj.shape[0]

    # --- 解析解部分保持不变 ---
    if analytical:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
            
        A = sigma * G_adj
        M = A + sp.identity(N)
        b = np.array(A.sum(axis=-1))
        
        # 使用稀疏求解器
        Psi = sp.linalg.spsolve(M, b)
        return True, Psi
        
    # --- 迭代解部分 ---
    else:
        if sigma == -1:
            sigma, _ = optimal_sigma(G_adj, analytical=False)
        pGAdj = sigma * G_adj.astype(np.float64)
        row_sums = np.array(pGAdj.sum(axis=1)).flatten()
        Psi = np.ones(N, dtype=np.float64) / N
        dt = np.float64(dt)
        boundary = epsilon * N * dt # Boundary for tempVal L1-norm

        start_time = time.time()
        if anderson_freq > 0:
            # 使用列表来动态管理历史记录，更符合你的原始逻辑
            x_history = []
            g_history = [] # g(x) = f(x)
            m = anderson_memory
            
       
        def fixed_point_function(psi_k):
            influence_decay = pGAdj @ psi_k
            influence_term = row_sums - influence_decay
            tempVal = (influence_term - psi_k) * dt
            return psi_k + tempVal.real
        maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)
        j = 0
        # --- 主迭代循环 ---
        for i in range(maxIter):
            
            # --- Step 1: 执行标准的DomiRank迭代步骤 ---
            f_psi = fixed_point_function(Psi)
            
            tempVal = f_psi - Psi # 这就是你的 tempVal
            if i % checkStep == 0 and i > 0:
                maxVals[j] = tempVal.max()
                l1_change = np.abs(tempVal).sum()
                if l1_change < boundary:
                    print(f"迭代在第 {i} 次收敛。")
                    break
                if j > 0:
                        if j > 1 and maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2] :#and maxVals[j-2] > maxVals[j-3]:
                            print("发散")
                            return False, Psi
                j+=1
            Psi = f_psi

            # --- Step 2: 周期性地应用Anderson加速 ---
            if anderson_freq > 0 and i > 0 and i % anderson_freq == 0:
                if len(x_history) == m:
                    x_history.pop(0)
                    g_history.pop(0)
                prev_psi = Psi - tempVal
                x_history.append(prev_psi.copy())
                g_history.append(Psi.copy()) # Psi 已经是 f(prev_psi) 了
                res_history = [g - x for g, x in zip(g_history, x_history)]
                G_k = np.array(res_history).T
                
                current_residual = res_history[-1]

                try:
                    G_k_T_G_k = G_k.T @ G_k
                    regularization = 1e-9
                    identity = np.eye(G_k_T_G_k.shape[0])
                    
                    target = G_k.T @ current_residual
                    gamma = np.linalg.solve(G_k_T_G_k + regularization * identity, target)
                    
                    accelerated_psi = g_history[-1] - G_k @ gamma
                    
                    Psi = accelerated_psi

                except np.linalg.LinAlgError:
                    # 如果求解失败，则不进行加速，继续标准迭代
                    pass

        time_improve = time.time() - start_time
        anderson_status = f"Anderson (freq={anderson_freq})" if anderson_freq > 0 else "Standard"
        print(f"总时间 ({anderson_status}): {time_improve:.4f}秒")
        
        if i == maxIter - 1:
            print("警告: 达到最大迭代次数，可能未完全收敛。")
            
        return True, Psi


import scipy as sp 
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 50, dt = 0.1, epsilon = 1e-7, maxIter = 1000, checkStep = 10,compression_plan = None,precomputed_mask=None,omega = None,anderson_enabled = None):
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
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep,None,compression_plan,precomputed_mask,omega,anderson_enabled)[0]:
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


###############################################################最优sigma初始求解方法###########################
import multiprocessing as mp
import numpy as np

# --- 1. 定义全局变量 ---
# 我们将在这里存放共享的大矩阵
shared_spArray = None

def init_worker(spArray_from_main):
    global shared_spArray
    shared_spArray = spArray_from_main

def process_iteration_shared(sigma, analytical, maxIter, checkStep, dt, epsilon, sampling,compression_plan,precomputed_mask,omega):
    """
    【内存安全版】的并行任务函数。
    它不再接收 spArray 作为参数，而是直接从全局变量中读取。
    """
    global shared_spArray # 声明我们要使用全局的矩阵
    
    try:
        tf, domiDist = domirank(shared_spArray, analytical=analytical, sigma=sigma, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep,compression_plan=compression_plan,precomputed_mask=precomputed_mask,omega=omega)
        if tf:
            domiAttack = generate_attack(domiDist)
            ourTempAttack = network_attack_sampled(shared_spArray, domiAttack, sampling=sampling)
            num_to_sum = int(len(ourTempAttack['components']) * 1.0)
            finalError = np.array(ourTempAttack['components'])[:num_to_sum].sum()
            print(f"Sigma={sigma:.12f} 计算完成, AUC={finalError:.4f}")
        else:
            finalError = 450.0
            print(f"Sigma={sigma:.12f} 未收敛。")
        return (sigma, finalError, tf) # tf 是收敛标志
        
    except Exception as e:
        print(f"Sigma={sigma:.6f} 的计算失败: {e}")
        return None

def optimal_sigma(spArray, analytical=True, endVal=0, startval=0.0000001, iterationNo=100, dt=1e-5, epsilon=1e-7, maxIter=1000, checkStep=5, maxDepth=100, sampling=0,compression_plan = None,precomputed_mask=None,omega = None):
    """
    【内存安全版】的最优 sigma 搜索函数。
    """
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth=maxDepth, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

    endval = -0.9999 /(endVal*10)
    tempRange = np.linspace(startval, endval, iterationNo)
    max_processes = 6
    print(f"启动进程池，使用 {max_processes} 个工作进程...")
    tasks = [(sigma, analytical, maxIter, checkStep, dt, epsilon, sampling,compression_plan,precomputed_mask,omega) for sigma in tempRange]

    # 2. 创建进程池，并通过 initializer 和 initargs 将大矩阵安全地“注入”到每个子进程
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