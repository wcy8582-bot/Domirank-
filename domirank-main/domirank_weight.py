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
        noComponent, lenComponent = scipy.sparse.csgraph.connected_components(G, directed = True, connection = connection_type, return_labels = True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')



def get_lcc_metrics(graph_or_matrix):
    
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    # --- 路径 A: 输入是 NetworkX Graph ---
    if isinstance(graph_or_matrix, nx.Graph):
        from scipy.sparse import csgraph
        if graph_or_matrix.number_of_edges() == 0:
            return 1, 0, 0.0
        try:
            adj_matrix = nx.to_scipy_sparse_array(graph_or_matrix, weight='weight', format='csr')
        except Exception as e:
            print(f"警告: 转换为稀疏矩阵失败: {e}。将回退到慢速的 NetworkX 方法。")
            lcc_nodes = max(nx.connected_components(graph_or_matrix), key=len)
            lcc_subgraph = graph_or_matrix.subgraph(lcc_nodes)
            return lcc_subgraph.number_of_nodes(), lcc_subgraph.number_of_edges(), lcc_subgraph.size(weight='weight')
        n_components, labels = csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        if n_components == 0:
            return 0, 0, 0.0
        if labels.size == 0: 
             return 1, 0, 0.0
        component_sizes = np.bincount(labels)
        lcc_id = np.argmax(component_sizes)
        node_count = component_sizes[lcc_id]
        lcc_nodes_indices = np.where(labels == lcc_id)[0]
        lcc_submatrix = adj_matrix[lcc_nodes_indices, :][:, lcc_nodes_indices]
        link_count = lcc_submatrix.nnz / 2.0
        weighted_flow = lcc_submatrix.sum() / 2.0
        return int(node_count), int(link_count), weighted_flow
    # if isinstance(graph_or_matrix, nx.Graph):
    #     G = graph_or_matrix
    #     if G.number_of_nodes() == 0: return 0, 0, 0.0
    #     if G.number_of_edges() == 0: return G.number_of_nodes(), 0, 0.0
    #     try:
    #         lcc_nodes = max(nx.connected_components(G), key=len)
    #         lcc_subgraph = G.subgraph(lcc_nodes)
    #         return lcc_subgraph.number_of_nodes(), lcc_subgraph.number_of_edges(), lcc_subgraph.size(weight='weight')
    #     except ValueError:
    #         return 0, 0, 0.0


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
        if labels.size == 0: 
             return 1, 0, 0.0
             
        component_sizes = np.bincount(labels)
        lcc_label = np.argmax(component_sizes)
        lcc_node_count = component_sizes[lcc_label]
        if lcc_node_count == 0 and num_nodes > 0:
            return 1, 0, 0.0

        lcc_indices = np.where(labels == lcc_label)[0]
        lcc_submatrix = GAdj[lcc_indices, :][:, lcc_indices]
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
        
        diag_matrix = sp.diags(mask.astype(int), offsets=0, format='csr', dtype=np.float64)
        
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
        attack_strategy_q = attack_strategy_q[sampling:]
        GAdj_copy = remove_node(GAdj_copy, batch_to_remove)
        
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




# ######## Beginning of domirank stuff! ####################

# def domirank(G, analytical = False, sigma = -1, dt = 0.1, epsilon = 1e-6, maxIter = 500, checkStep = 10):
#     '''
#     G is the input graph as a (preferably) sparse array.
#     This solves the dynamical equation presented in the Paper: "DomiRank Centrality: revealing structural fragility of
#     complex networks via node dominance" and yields the following output: bool, DomiRankCentrality
#     Here, sigma needs to be chosen a priori.
#     dt determines the step size, usually, 0.1 is sufficiently fine for most networks (could cause issues for networks
#     with an extremely high degree, but has never failed me!)
#     maxIter is the depth that you are searching with in case you don't converge or diverge before that.
#     Checkstep is the amount of steps that you go before checking if you have converged or diverged.
    
    
#     This algorithm scales with O(m) where m is the links in your sparse array.
#     '''
    # if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
    #     G_nx = G
    #     G = nx.to_scipy_sparse_array(G,weight ="weight") #convert to scipy sparse if it is a graph 
    # else:
    #     G_nx = nx.from_scipy_sparse_array(G, create_using=nx.Graph)
    #     G = G.copy()
#     N = G.shape[0]
#     if N == 0: return True, np.array([])
#     if analytical == False:
#         if sigma == -1:
#             sigma, _ = optimal_sigma(G, analytical = False, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
#         pGAdj = sigma*G.astype(np.float64)
        # pagerank_scores_dict = nx.pagerank(G_nx, alpha=0.85)
        # pagerank_scores = np.array([pagerank_scores_dict.get(i, 0) for i in range(N)])
        # Psi = pagerank_scores / pagerank_scores.sum()
#         Psi = np.ones(N).astype(np.float64) / N

#         maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64) 
#         j = 0
#         boundary = epsilon*pGAdj.shape[0]*dt
#         for i in range(maxIter):
            # tempVal = ((pGAdj @ (1 - Psi)) - Psi) * dt
            # Psi += tempVal.real
#             if i % 5 == 0:
#                 row_sums = np.array(pGAdj.sum(axis=1)).flatten()
#                 term1 = row_sums - (pGAdj @ Psi)
#             else:
#                 term1 = pGAdj @ (1 - Psi)
#             tempVal = (term1 - Psi) * dt
#             Psi += tempVal.real
            # if i% checkStep == 0:
            #     if np.abs(tempVal).sum() < boundary:
            #         print(f"在第 {i} 步收敛。")
            #         break
                # maxVals[j] = tempVal.max()
                # if i == 0:
                #     initialChange = maxVals[j]
                # if j > 0:
                #     if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                #         return False, Psi
                # j+=1
#         return True, Psi
#     else:
#         if sigma == -1:
#             sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
#         Psi = sp.sparse.linalg.spsolve(sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
#         return True, Psi


def domirank(G, analytical=False, sigma=-1, dt=0.1, epsilon=1e-8, maxIter=3000, checkStep=5, 
                                          mode='standard', alpha=0.1, output_filename=None): 
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

    if analytical:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        Psi = sp.linalg.spsolve(sigma*G + sp.identity(G.shape[0]), sigma*G.sum(axis=-1))
        return True, Psi
    else:
        if sigma == -1:
            sigma, _ = optimal_sigma(G_adj, analytical=False)
        pGAdj = sigma * G_adj.astype(np.float64)
        all_indices = np.arange(N)
        row_sums = np.array(pGAdj.sum(axis=1)).flatten()
        psi_sum = row_sums.sum()
        # Psi = row_sums / psi_sum if psi_sum > 0 else np.ones(N, dtype=np.float64)
        Psi = np.ones(N, dtype=np.float64) / N
        j = 0
        dt = np.float64(dt)
        boundary = epsilon * N * dt 
        maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)
        start_time = time.time()
        pGAdj_sliced_inefficiently = pGAdj[all_indices, :][:, all_indices]
        for i in range(maxIter):
            influence_decay = pGAdj_sliced_inefficiently @ Psi
            influence_term = row_sums - influence_decay
            tempVal = (influence_term - Psi) * dt
            # num_sub_cycles = 20
            # sub_step = tempVal.real / num_sub_cycles
            # for _ in range(num_sub_cycles):
            #     Psi += sub_step
            Psi += tempVal.real
            if i % checkStep == 0 and i != 0:
                if np.abs(tempVal).sum() < boundary:
                    # print("最大迭代次数:",i)
                    break
                maxVals[j] = tempVal.max()
                max_change = np.abs(tempVal).max()
                if not np.isfinite(max_change):
                    print(f"错误：在第 {i} 次迭代时计算结果发散 (NaN/inf)。")
                    return False, Psi
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if j > 1 and maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2] :#and maxVals[j-2] > maxVals[j-3]:
                        print("发散")
                        return False, Psi
                j+=1
        time_improve = time.time() - start_time
        print("总时间:",time_improve)
        return True, Psi
    
    # else:
    #     if sigma == -1:
    #         print("警告: optimal_sigma 函数未提供，使用默认 sigma=0.1。")
    #         sigma = 0.1

    #     # --- 算法初始化部分保持不变 ---
    #     pGAdj = sigma * G_adj.astype(np.float64)
    #     row_sums = np.array(pGAdj.sum(axis=1)).flatten()
    #     psi_sum = row_sums.sum()
    #     Psi = row_sums / psi_sum if psi_sum > 0 else np.ones(N, dtype=np.float64)
    #     j = 0
    #     dt = np.float64(dt)
    #     boundary = epsilon * N * dt
    #     maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)
    #     start_time = time.time()
        
    #     converged_nodes_mask = np.zeros(N, dtype=bool)
    #     node_convergence_threshold = 1e-6

    #     # --- 核心迭代循环 ---
    #     for i in range(maxIter):
    #         influence_decay = pGAdj @ Psi
    #         influence_term = row_sums - influence_decay
    #         tempVal = (influence_term - Psi) * dt
            
    #         # 更新收敛节点掩码
    #         newly_converged_mask = (np.abs(tempVal) < node_convergence_threshold) & (~converged_nodes_mask)
    #         if np.any(newly_converged_mask):
    #             converged_nodes_mask[newly_converged_mask] = True
            
    #         Psi += tempVal.real
    #         # if i == 100 and output_filename:
    #         #     print(f"\n在第 {i} 次迭代，正在将已冻结节点数据写入文件: {output_filename}")
    #         #     frozen_node_indices = np.where(converged_nodes_mask)[0]
    #         #     if frozen_node_indices.size > 0:
    #         #         frozen_node_scores = Psi[frozen_node_indices]
    #         #         with open(output_filename, 'w') as f:
    #         #             f.write("# Node_ID Score_at_100_iterations\n")
    #         #             for idx, score in zip(frozen_node_indices, frozen_node_scores):
    #         #                 node_id = node_list[idx] if is_graph_input else idx
    #         #                 f.write(f"{node_id} {score}\n")
    #         #         print(f"成功写入 {len(frozen_node_indices)} 个已冻结节点的数据。\n")
    #         #     else:
    #         #         print("在第 100 次迭代时，没有节点被冻结。\n")
    #         if i % checkStep == 0 and i != 0:
    #             # if i  % 100 == 0:
    #             #     print(f"    迭代 {i}: 总收敛节点数: {np.sum(converged_nodes_mask)} / {N}")
    #             if np.abs(tempVal).sum() < boundary:
    #                 print("已满足全局收敛条件，停止迭代。")
    #                 print("最终迭代次数:",i)
    #                 break
    #             maxVals[j] = tempVal.max()
    #             max_change = np.abs(tempVal).max()
    #             if not np.isfinite(max_change):
    #                 print(f"错误：在第 {i} 次迭代时计算结果发散 (NaN/inf)。")
    #                 return False, Psi
    #             if i == 0:
    #                 initialChange = maxVals[j]
    #             if j > 0:
    #                 if j > 1 and maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2] :#and maxVals[j-2] > maxVals[j-3]:
    #                     print("发散")
    #                     return False, Psi
    #             j+=1
    #     time_improve = time.time() - start_time
    #     print("总时间:",time_improve)
    #     return True, Psi
    

import scipy as sp 
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 50, dt = 0.1, epsilon = 1e-9, maxIter = 1500, checkStep = 10):
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




############## This section is for finding the optimal sigma ######################

import multiprocessing as mp
import numpy as np

# --- 1. 定义全局变量 ---
# 我们将在这里存放共享的大矩阵
shared_spArray = None

def init_worker(spArray_from_main):
    global shared_spArray
    shared_spArray = spArray_from_main

def process_iteration_shared(sigma, analytical, maxIter, checkStep, dt, epsilon, sampling):
    """
    【内存安全版】的并行任务函数。
    它不再接收 spArray 作为参数，而是直接从全局变量中读取。
    """
    global shared_spArray # 声明我们要使用全局的矩阵
    
    try:
        tf, domiDist = domirank(shared_spArray, analytical=analytical, sigma=sigma, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
        if tf:
            domiAttack = generate_attack(domiDist)
            ourTempAttack = network_attack_sampled(shared_spArray, domiAttack, sampling=sampling)
            num_to_sum = int(len(ourTempAttack['links']) * 1.0)#components links
            finalError = np.array(ourTempAttack['links'])[:num_to_sum].sum()#components
            print(f"Sigma={sigma:.6f} 计算完成, AUC={finalError:.4f}")
        else:
            finalError = 450.0
            print(f"Sigma={sigma:.6f} 未收敛。")

        # 返回更丰富的信息
        return (sigma, finalError, tf) # tf 是收敛标志
        
    except Exception as e:
        print(f"Sigma={sigma:.6f} 的计算失败: {e}")
        return None

def optimal_sigma(spArray, analytical=True, endVal=0, startval=0.0000001, iterationNo=50, dt=1e-4, epsilon=1e-9, maxIter=1500, checkStep=5, maxDepth=100, sampling=0):
    """
    【内存安全版】的最优 sigma 搜索函数。
    """
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth=maxDepth, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

    endval = -0.9999 / endVal
    tempRange = np.linspace(startval, endval, iterationNo)
    
    max_processes = 5
    print(f"启动进程池，使用 {max_processes} 个工作进程...")

    # --- 关键改动在这里 ---
    # 1. 准备传递给子进程的、不包含大矩阵的参数
    tasks = [(sigma, analytical, maxIter, checkStep, dt, epsilon, sampling) for sigma in tempRange]

    # 2. 创建进程池，并通过 initializer 和 initargs 将大矩阵安全地“注入”到每个子进程
    with mp.Pool(processes=max_processes, 
                 initializer=init_worker, 
                 initargs=(spArray,)) as pool:
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