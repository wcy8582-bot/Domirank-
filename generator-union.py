import itertools
from pathlib import Path
import pickle
import random
import os
import igraph as ig
import leidenalg as la
from multiprocessing import Pool, cpu_count
from itertools import combinations
import shutil 
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, 'build')

import time
import gc
from typing import Tuple, Dict, Any


import itertools
import numpy as np
import networkx as nx
from typing import Dict, Any
from tqdm import tqdm


# def calculate_custom_extensibility_FINAL(G: nx.Graph) -> Dict[Any, float]:
#     print("--- [一阶 + 二阶 自定义版延展率计算开始] ---")
    
#     # --- 步骤 1: 预计算 ---
#     print("  -> 步骤 1/3: 正在预计算邻居和度数...")
#     neighbors_dict = {
#         node: set(G.neighbors(node))
#         for node in tqdm(G.nodes(), desc="  - 预计算邻居")
#     }
#     degrees_dict = {
#         node: len(neighbors_dict[node])
#         for node in G.nodes()
#     }

#     extensibility_scores = {}

#     # 一阶和二阶权重：x + x^2 = 1
#     x = (np.sqrt(5) - 1) / 2.0
#     weight_1 = x
#     weight_2 = x ** 2

#     print(f"  -> 一阶权重 x = {weight_1:.6f}")
#     print(f"  -> 二阶权重 x^2 = {weight_2:.6f}")
#     print(f"  -> 权重和 = {weight_1 + weight_2:.6f}")
    
#     # --- 步骤 2: 逐节点计算 ---
#     print("  -> 步骤 2/3: 正在逐节点计算延展性分数...")

#     for u in tqdm(G.nodes(), desc="  - 计算节点分数", total=G.number_of_nodes()):
        
#         neighbors_of_u = neighbors_dict.get(u, set())
#         deg_u = degrees_dict.get(u, 0)
        
#         # =====================================================
#         # 一阶邻居延展性计算：保持你的原始逻辑不变
#         # =====================================================
#         if deg_u < 2:
#             first_order_score = 1.0
#         else:
#             total_connectivity_strength = 0

#             for v, w in itertools.combinations(neighbors_of_u, 2):
#                 common_neighbors_count = len((neighbors_dict[v] & neighbors_dict[w]) - {u})
#                 direct_edge_bonus = 1 if w in neighbors_dict.get(v, set()) else 0
#                 pair_strength = common_neighbors_count + direct_edge_bonus
#                 total_connectivity_strength += pair_strength
            
#             num_neighbor_pairs = (deg_u - 1)

#             neighbor_degrees = [
#                 degrees_dict.get(v, 0)
#                 for v in neighbors_of_u
#             ]

#             avg_neighbor_degree = np.sum(neighbor_degrees) if neighbor_degrees else 0.0
#             avg_max_pair_strength = avg_neighbor_degree

#             denominator = num_neighbor_pairs * avg_max_pair_strength

#             if denominator == 0:
#                 cohesion = 0.0
#             else:
#                 cohesion = total_connectivity_strength / denominator

#             first_order_score = 1.0 - cohesion

#         # =====================================================
#         # 二阶邻居延展性计算
#         # =====================================================
#         # u 的二阶邻居：
#         # u 的一阶邻居的邻居，去掉 u 自己，去掉 u 的一阶邻居
#         second_order_neighbors = set()

#         for v in neighbors_of_u:
#             second_order_neighbors.update(neighbors_dict.get(v, set()))

#         second_order_neighbors.discard(u)
#         second_order_neighbors -= neighbors_of_u

#         second_order_count = len(second_order_neighbors)

#         if second_order_count < 2:
#             # 二阶邻居不足两个，无法形成二阶邻居对
#             second_order_raw_score = 1.0
#         else:
#             total_second_order_connectivity_strength = 0

#             # 分子：
#             # u 的二阶邻居中，所有节点对的公共邻居数量之和
#             for p, q in itertools.combinations(second_order_neighbors, 2):
#                 common_neighbors_count = len(neighbors_dict[p] & neighbors_dict[q])+1
#                 total_second_order_connectivity_strength += common_neighbors_count

#             # 分母：
#             # u 的二阶邻居数量 * u 的二阶邻居的度数总和
#             second_order_degrees = [
#                 degrees_dict.get(node, 0)
#                 for node in second_order_neighbors
#             ]

#             second_order_degree_sum = (
#                 np.sum(second_order_degrees)
#                 if second_order_degrees
#                 else 0.0
#             )

#             second_order_denominator = (second_order_count) * second_order_degree_sum

#             if second_order_denominator == 0:
#                 second_order_cohesion = 0.0
#             else:
#                 second_order_cohesion = (
#                     total_second_order_connectivity_strength
#                     / second_order_denominator
#                 )

#             second_order_raw_score = 1.0 - second_order_cohesion

#         # 你要求：
#         # 二阶节点分数还要乘以一阶节点分数
#         second_order_score = first_order_score * second_order_raw_score

#         # =====================================================
#         # 一阶 + 二阶 加权融合
#         # =====================================================
#         final_score = (
#             weight_1 * first_order_score
#              + weight_2 * second_order_score
#         )

#         extensibility_scores[u] = final_score

#     print("  -> 步骤 3/3: 所有节点分数计算完毕。")
#     return extensibility_scores

from collections import Counter
def calculate_custom_extensibility_FINAL(G: nx.Graph) -> Dict[Any, float]:
    print("--- [一阶 + 二阶 自定义版延展率计算开始] ---")
    
    # --- 步骤 1: 预计算 ---
    print("  -> 步骤 1/3: 正在预计算邻居和度数...")
    neighbors_dict = {
        node: set(G.neighbors(node))
        for node in tqdm(G.nodes(), desc="  - 预计算邻居")
    }
    degrees_dict = {
        node: len(neighbors_dict[node])
        for node in G.nodes()
    }

    extensibility_scores = {}

    x = (np.sqrt(5) - 1) / 2.0
    weight_1 = x
    weight_2 = x ** 2

    print(f"  -> 一阶权重 x = {weight_1:.6f}")
    print(f"  -> 二阶权重 x^2 = {weight_2:.6f}")
    print(f"  -> 权重和 = {weight_1 + weight_2:.6f}")
    
    # --- 步骤 2: 逐节点计算 ---
    print("  -> 步骤 2/3: 正在逐节点计算延展性分数...")

    for u in tqdm(G.nodes(), desc="  - 计算节点分数", total=G.number_of_nodes()):
        
        neighbors_of_u = neighbors_dict.get(u, set())
        deg_u = degrees_dict.get(u, 0)
        
        # =====================================================
        # 一阶邻居延展性计算：完全不变
        # =====================================================
        if deg_u < 2:
            first_order_score = 1.0
        else:
            total_connectivity_strength = 0

            for v, w in itertools.combinations(neighbors_of_u, 2):
                common_neighbors_count = len((neighbors_dict[v] & neighbors_dict[w]) - {u})
                direct_edge_bonus = 1 if w in neighbors_dict.get(v, set()) else 0
                pair_strength = common_neighbors_count + direct_edge_bonus
                total_connectivity_strength += pair_strength
            
            num_neighbor_pairs = (deg_u - 1)

            neighbor_degrees = [
                degrees_dict.get(v, 0)
                for v in neighbors_of_u
            ]

            avg_neighbor_degree = np.sum(neighbor_degrees) if neighbor_degrees else 0.0
            avg_max_pair_strength = avg_neighbor_degree

            denominator = num_neighbor_pairs * avg_max_pair_strength

            if denominator == 0:
                cohesion = 0.0
            else:
                cohesion = total_connectivity_strength / denominator

            first_order_score = 1.0 - cohesion

        # =====================================================
        # 二阶邻居延展性计算（集合构建部分不变）
        # =====================================================
        second_order_neighbors = set()

        for v in neighbors_of_u:
            second_order_neighbors.update(neighbors_dict.get(v, set()))

        second_order_neighbors.discard(u)
        second_order_neighbors -= neighbors_of_u

        second_order_count = len(second_order_neighbors)

        if second_order_count < 2:
            second_order_raw_score = 1.0
        else:
            common_counter = Counter()
            for p in second_order_neighbors:
                for r in neighbors_dict.get(p, ()): 
                    common_counter[r] += 1

            pair_common_sum = 0
            for c in common_counter.values():
                if c >= 2:
                    pair_common_sum += c * (c - 1) // 2

            num_pairs = second_order_count * (second_order_count - 1) // 2
            total_second_order_connectivity_strength = pair_common_sum + num_pairs

            # 分母：不变
            second_order_degrees = [
                degrees_dict.get(node, 0)
                for node in second_order_neighbors
            ]

            second_order_degree_sum = (
                np.sum(second_order_degrees)
                if second_order_degrees
                else 0.0
            )

            second_order_denominator = (second_order_count) * second_order_degree_sum

            if second_order_denominator == 0:
                second_order_cohesion = 0.0
            else:
                second_order_cohesion = (
                    total_second_order_connectivity_strength
                    / second_order_denominator
                )

            second_order_raw_score = 1.0 - second_order_cohesion

        second_order_score = first_order_score * second_order_raw_score
        final_score = (
            weight_1 * first_order_score
             + weight_2 * second_order_score
        )

        extensibility_scores[u] = final_score

    print("  -> 步骤 3/3: 所有节点分数计算完毕。")
    return extensibility_scores


def precompute_inter_community_edges(G, partition):
    """
    【Pickle 安全版】
    预计算图中每对社群之间的连接总数。
    使用普通的 dict 以确保结果可以被安全地序列化 (pickle)。

    Args:
        G (nx.Graph): 图。
        partition (dict): 节点到社群ID的映射。

    Returns:
        dict: 一个嵌套字典，community_edge_counts[c1][c2] 存储了社群c1和c2之间的边数。
    """
    community_edge_counts = {}  # 使用普通字典
    
    for u, v in G.edges():
        c1 = partition.get(u)
        c2 = partition.get(v)
        
        # 确保 c1 和 c2 都有效，并且是跨社群的边
        if c1 is not None and c2 is not None and c1 != c2:
            # 为了统计方便，我们让 c1 < c2，避免重复计算 (c1, c2) 和 (c2, c1)
            if c1 > c2:
                c1, c2 = c2, c1
            outer_dict = community_edge_counts.setdefault(c1, {})
            
            # 2. 确保内层字典有 c2 这个键，如果没，则将其值设置为 0
            # 3. 然后再执行 += 1 操作
            outer_dict[c2] = outer_dict.get(c2, 0) + 1
            # --- 修改结束 ---
            
    return community_edge_counts


def calculate_inter_community_uniqueness(
    u, v, 
    partition, 
    community_props, 
    community_edge_counts
) -> float:
    comm_u = partition.get(u)
    comm_v = partition.get(v)
    if comm_u is None or comm_v is None or comm_u == comm_v:
        return 0.1
    if comm_u > comm_v:
        comm_u, comm_v = comm_v, comm_u
    inter_community_edge_count = community_edge_counts.get(comm_u, {}).get(comm_v, 0)
    if inter_community_edge_count <= 1:
        return 1.0

    size_u = community_props.get(comm_u, {}).get('size', 1)
    size_v = community_props.get(comm_v, {}).get('size', 1)

    avg_size = np.sqrt(size_u * size_v)
    alpha = 1.0 / np.log1p(avg_size + 1) 
    max_possible_edges_scaled = ((size_u) * (size_v)) #** alpha

    if max_possible_edges_scaled <= 1:
        return 1.0
    redundancy = (inter_community_edge_count - 1) / (max_possible_edges_scaled )
    uniqueness = np.log1p( (1 / redundancy))
    # uniqueness =  np.clip(1.0 - redundancy, 0, 1)
    return uniqueness


def calculate_uniqueness_score(G: nx.Graph, u, v, common_neighbors: list) -> float:
    """
    计算边的唯一性/非冗余性分数（即 1 - redundancy）。
    包含性能优化的 cutoff 逻辑。
    Args:
        G (nx.Graph): 完整的图。
        u, v: 边的两个端点。
        common_neighbors (list): u 和 v 的共同邻居列表。

    Returns:
        float: 一个在 [0, 1] 范围内的唯一性分数。
    """
    COMMON_NEIGHBOR_CUTOFF = 300000 # 保留性能优化
    common_neighbors_count = len(common_neighbors)
    skipped_edges_count = 0
    if common_neighbors_count <=1:
        return 1.0
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    avg_size = np.sqrt(deg_u * deg_v)
    alpha = 1.0 / np.log1p(avg_size + 1) 
    if common_neighbors_count > 1:
                if common_neighbors_count <= COMMON_NEIGHBOR_CUTOFF:
                    internal_edges = 0
                    for c1, c2 in combinations(common_neighbors, 2):
                        if G.has_edge(c1, c2):
                            internal_edges += 1
                    max_possible_edges = common_neighbors_count * (common_neighbors_count - 1) / 2.0
                    if max_possible_edges > 0:
                        redundancy = internal_edges / max(internal_edges, max_possible_edges)
                        penalty_exponent = 1.0
                        # redundancy_penalty_factor = np.log1p(1 -  redundancy)
                        redundancy_penalty_factor = 1 -  redundancy
                else:
                    # redundancy_penalty_factor = 0.01
                    skipped_edges_count += 1
                    SAMPLE_SIZE = 150
                    internal_edges_in_sample = 0
                    if not isinstance(common_neighbors, (list, tuple)):
                        common_neighbors = list(common_neighbors)
                        
                    # 进行随机抽样
                    for _ in range(SAMPLE_SIZE):
                        c1, c2 = random.sample(common_neighbors, 2)
                        if G.has_edge(c1, c2):
                            internal_edges_in_sample += 1
                    estimated_redundancy = internal_edges_in_sample / SAMPLE_SIZE
                    redundancy_penalty_factor = 1 - estimated_redundancy
    return redundancy_penalty_factor
    


import networkx as nx
import community as community_louvain

def precompute_community_and_roles(G):
    """
    预计算社群信息和每个节点的角色。

    Returns:
        tuple: (
            partition (dict): {node: community_id},
            community_centralities (dict): {node: in_community_degree_centrality}
        )
    """
    print("步骤1: 预计算社群结构...")
    partition = community_louvain.best_partition(G)
    community_props = {}
    for node, comm_id in partition.items():
        if comm_id not in community_props:
            community_props[comm_id] = {'nodes': [], 'size': 0}
        community_props[comm_id]['nodes'].append(node)
        community_props[comm_id]['size'] += 1

    print(f"  -> 社群结构和属性计算完毕。发现 {len(community_props)} 个社群。")
    return partition, community_props




def _relabel_edges_chunk(chunk_data):
    """
    【工作函数】由每个并行进程执行。
    处理一小块边，并使用全局映射对其进行重标记。
    """
    edges_chunk, mapping = chunk_data
    return [(mapping[u], mapping[v]) for u, v in edges_chunk]

def _process_results_chunk(chunk_data):
    """
    【工作函数】由每个并行进程执行。
    处理一小块 (igraph_vertex_id, community_id) 数据，并返回局部的结果。
    """
    results_chunk, reverse_mapping = chunk_data
    local_partition_dict = {}
    local_community_props = {}
    
    for i, comm_id in results_chunk:
        original_node = reverse_mapping[i]
        local_partition_dict[original_node] = comm_id
        
        if comm_id not in local_community_props:
            local_community_props[comm_id] = {'nodes': [], 'size': 0}
        local_community_props[comm_id]['nodes'].append(original_node)
        local_community_props[comm_id]['size'] += 1
        
    return local_partition_dict, local_community_props



def precompute_community_and_roles_accelerated(
    G: nx.Graph, 
    num_workers: int = -1
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
   
    print("--- 开始【终极并行加速版】社群预计算 ---")
    total_start_time = time.time()

    if num_workers == -1:
        num_workers = max(1, cpu_count() - 90)
    print(f"  -> 全局设置: 将使用 {num_workers} 个CPU核心进行并行处理。")

    # --- 步骤 1: 【并行化】高效图转换 ---
    print("\n步骤1/4: 正在【并行化】将 NetworkX 图转换为 igraph...")
    start_time = time.time()
    
    # 1a. 创建映射 (这步很快，无需并行)
    print("  - [1a] 创建节点映射...")
    all_nodes_list = list(G.nodes()) # 一次性获取所有节点
    sorted_nodes = sorted(all_nodes_list)
    mapping = {node: i for i, node in enumerate(sorted_nodes)}
    reverse_mapping = sorted_nodes # 反向映射现在就是一个列表
    del all_nodes_list, sorted_nodes
    gc.collect()

    # 1b. 【并行】重标记边列表
    print("  - [1b] 【并行】重标记边列表...")
    edges_original = list(G.edges())
    total_edges = len(edges_original)
    # 立即释放原始大图对象的内存
    del G
    gc.collect()


    chunksize = max(1, total_edges // (num_workers * 4))
    def generate_relabeled_edges(pool, tasks, total_chunks):
        pbar = tqdm(pool.imap_unordered(_relabel_edges_chunk, tasks),
                    total=total_chunks,
                    desc="  重标记进度 (流式)")
        for chunk_result in pbar:
            yield from chunk_result
    print("  - [1c] 创建空的igraph图对象...")
    num_vertices = len(reverse_mapping)
    G_ig = ig.Graph(n=num_vertices, directed=False) # 假设无向
    with Pool(num_workers) as pool:
        tasks = ((edges_original[i:i + chunksize], mapping) for i in range(0, total_edges, chunksize))
        total_chunks = (total_edges + chunksize - 1) // chunksize
        edge_generator = generate_relabeled_edges(pool, tasks, total_chunks)
        print("  - [1c] 正在从数据流创建igraph图...")
        G_ig.add_edges(edge_generator)
    del edges_original, mapping 
    gc.collect()
    
    print(f"  -> 步骤1总耗时: {time.time() - start_time:.2f} 秒。")
    print(f"  -> igraph 图: {G_ig.vcount()} 个顶点, {G_ig.ecount()} 条边。")

    # --- 步骤 2: 运行 Leiden 算法 (核心计算，保持串行) ---
    print("\n步骤2/4: 正在运行 Leiden 算法 (C++后端)...")
    print("  -> (此为计算密集型核心，不显示进度条，请耐心等待...)")
    start_time = time.time()
    partition_leiden = la.find_partition(G_ig, la.ModularityVertexPartition, seed=42)
    del G_ig
    gc.collect()
    print(f"  -> Leiden 算法运行完毕，耗时: {time.time() - start_time:.2f} 秒。")
    print(f"  -> 发现 {len(partition_leiden)} 个社群，模块度: {partition_leiden.modularity:.4f}。")

    # --- 步骤 3: 【并行化】结果转换与属性计算 ---
    print("\n步骤3/4: 正在【并行化】转换结果并计算社群属性...")
    start_time = time.time()
    
    membership_list = list(enumerate(partition_leiden.membership))
    total_nodes = len(membership_list)
    chunksize = max(1, total_nodes // (num_workers * 4))

    partition_dict = {}
    community_props = {}

    with Pool(num_workers) as pool:
        tasks = ((membership_list[i:i + chunksize], reverse_mapping) for i in range(0, total_nodes, chunksize))
        
        pbar = tqdm(pool.imap_unordered(_process_results_chunk, tasks),
                    total=(total_nodes + chunksize - 1) // chunksize,
                    desc="  处理进度")
        
        for local_part, local_props in pbar:
            partition_dict.update(local_part)
            for comm_id, props in local_props.items():
                if comm_id not in community_props:
                    community_props[comm_id] = {'nodes': [], 'size': 0}
                community_props[comm_id]['nodes'].extend(props['nodes'])
                community_props[comm_id]['size'] += props['size']

    print(f"  -> 结果转换完成，耗时: {time.time() - start_time:.2f} 秒。")
    
    # --- 步骤 4: 最终检查与总结 ---
    print("\n步骤4/4: 最终检查与总结...")
    print(f"  -> partition 字典大小: {len(partition_dict)} (应等于节点数)")
    print(f"  -> community_props 字典大小: {len(community_props)} (应等于社群数)")
    
    total_end_time = time.time()
    print(f"\n--- 社群预计算总耗时: {total_end_time - total_start_time:.2f} 秒 ---")
    
    return partition_dict, community_props


def process_all_strategies_in_one_pass(
    G: nx.Graph,
    output_dir: Path,
    file_stem: str,
    extensibility_scores: dict,
    partition: dict,
    community_props: dict,
    inter_community_counts: dict
):
    print("\n--- [核心流程] 开始一体化、可恢复的流式处理 ---")
    
    # --- 1. 准备通用数据 ---
    print("  - 准备通用数据...")
    degrees = dict(G.degree())
    
    # 关键改动：将 G.edges() 转换为列表，这是实现断点续跑的基础
    edges_list = list(G.edges())
    num_edges = len(edges_list)
    strategy_names = ['shengshe','e-shengshe','u-shengshe']
    progress_file = output_dir / f"{file_stem}-progress.txt"
    start_index = 0  # 默认从第0条边开始

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            try:
                content = f.read().strip()
                if content:
                    start_index = int(content)
                    print(f"  -> 发现进度文件！将从第 {start_index}/{num_edges} 条边继续处理。")
            except (ValueError, IndexError):
                print(f"  -> 警告：进度文件 '{progress_file.name}' 格式错误或为空，将从头开始。")
                start_index = 0

    # --- 3. 文件处理：根据是否续跑，决定打开模式 ---
    output_files = {}
    file_handles = {}
    open_mode = 'w' if start_index == 0 else 'a'
    
    for strategy in strategy_names:
        output_path = output_dir / f"{file_stem}-{strategy}.txt"
        output_files[strategy] = output_path
        file_handles[strategy] = open(output_path, open_mode, encoding='utf-8')
    
    print(f"  - 已以 '{open_mode}' 模式打开 {len(file_handles)} 个输出文件。")
    print(f"\n  - 开始单次遍历图的边 (从 {start_index} 到 {num_edges})...")
    pbar = tqdm(range(start_index, num_edges), total=num_edges, initial=start_index, desc="  策略计算进度")
    checkpoint_interval = 10000  # 每处理10000条边，保存一次进度
    sleep_interval = 100        # 每处理100条边，休息一会
    sleep_duration = 0.001      # 休息1毫秒

    for i in pbar:
        u, v = edges_list[i]
        neighbors_u = G[u]
        neighbors_v = G[v]
        
        if len(neighbors_u) < len(neighbors_v):
            common_neighbors_count = sum(1 for nbr in neighbors_u if nbr in neighbors_v)
        else:
            common_neighbors_count = sum(1 for nbr in neighbors_v if nbr in neighbors_u) 
        _common_neighbors_list = None 
        k_u = degrees[u]
        k_v = degrees[v]
        union_size = k_u + k_v - common_neighbors_count
        local_bridge_term = 1 - (common_neighbors_count / (union_size + 1))
        amplified_bridge_term_sq = local_bridge_term**2

        s_ext_u = extensibility_scores.get(u, 0)
        s_ext_v = extensibility_scores.get(v, 0)
        extensibility_term = s_ext_u * s_ext_v 
        if _common_neighbors_list is None:
            _common_neighbors_list = list(nx.common_neighbors(G, u, v))
        
        
        
        is_bridge = 1.0 if partition[u] != partition[v] else 0.0
        uniqueness_sheqv = 0.0
        if is_bridge > 0:
            uniqueness_sheqv =  calculate_inter_community_uniqueness(u, v, partition, community_props, inter_community_counts)
        else:
            uniqueness_sheqv = calculate_uniqueness_score(G, u, v, _common_neighbors_list)
        base_score_uniqueness = uniqueness_sheqv
        weight_uniqueness = base_score_uniqueness
        file_handles['u-shengshe'].write(
            f"{u} {v} {weight_uniqueness:.6f}\n"
        )

        # 2. 只使用 extensibility_term 的权重
        base_score_extensibility = extensibility_term
        weight_extensibility = base_score_extensibility
        file_handles['e-shengshe'].write(
            f"{u} {v} {weight_extensibility:.6f}\n"
        )
        base_score_shengshe =  uniqueness_sheqv * ( (extensibility_term)**0.5)
        weight_shengshe = base_score_shengshe
        file_handles['shengshe'].write(f"{u} {v} {weight_shengshe:.6f}\n")
        if (i + 1) % checkpoint_interval == 0:
            for handle in file_handles.values():
                handle.flush()
            with open(progress_file, 'w') as f:
                f.write(str(i + 1))
                f.flush()
                os.fsync(f.fileno()) 
            pbar.set_postfix_str("进度已保存!")
        if (i + 1) % sleep_interval == 0:
            time.sleep(sleep_duration)
    pbar.close()
    print("\n  - 所有边处理完毕。正在关闭文件...")
    for strategy, handle in file_handles.items():
        handle.close()
        print(f"    - 文件 '{output_files[strategy].name}' 已保存。")
    if progress_file.exists():
        os.remove(progress_file)
        print("  - 任务成功完成，已清理进度文件。")
        
    print("--- ✓ 所有策略处理完成！ ---")

if __name__ == '__main__':

    # ============================================================
    # [阶段 0] 配置：一次只处理一张图，防止多个大图同时进内存
    # ============================================================

    input_graph_paths = [
        '/home/wcy/domirank/domirank/Data/road-1500000/road.txt',
    ]

    # ============================================================
    # 工具函数 1：安全 pickle 写入
    # ============================================================

    def atomic_pickle_dump(obj, file_path):
        """
        以原子方式将对象 pickle 到文件，防止写入中断导致文件损坏。
       """
        temp_file_path = Path(str(file_path) + ".tmp")
        try:
            with open(temp_file_path, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.move(str(temp_file_path), str(file_path))
        except Exception as e:
            print(f"  -> 警告：原子写入失败: {e}")
            if temp_file_path.exists():
                os.remove(temp_file_path)
            raise

    # ============================================================
    # 工具函数 2：安全 pickle 读取
    # ============================================================

    def safe_pickle_load(file_path, cache_name):
        """
        安全读取缓存。
        如果缓存损坏，自动删除，返回 None。
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            print(f"     - ✓ {cache_name} 加载完毕。")
            return obj

        except (EOFError, pickle.UnpicklingError, OSError) as e:
            print(f"     - 警告：{cache_name} 缓存文件损坏，将删除并重新计算。错误: {e}")
            try:
                os.remove(file_path)
            except Exception:
                pass
            return None

    # ============================================================
    # 工具函数 3：轻量级大图加载
    # ============================================================

    def load_graph_light(input_graph_path, chunk_size=1_000_000):
        """
        轻量加载大图。

        重点：
        1. 不使用 nx.read_edgelist；
        2. 不保存 weight 属性；
        3. 每条边只保存 u-v；
        4. 分批 add_edges_from；
        5. 跳过注释行、空行、自环；
        6. 适配 2列/3列/多列边文件，只取前两列。

        这样比：
        nx.read_edgelist(..., data=[("weight", float)])
        省很多内存。
        """

        print(f"--- [轻量加载] 正在读取图文件: {input_graph_path} ---")

        start_time = time.time()
        G = nx.Graph()

        edge_buffer = []
        valid_edges = 0
        total_lines = 0

        with open(input_graph_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                total_lines += 1
                line = line.strip()

                if not line:
                    continue

                if line.startswith('#') or line.startswith('%'):
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    continue

                if u == v:
                    continue

                edge_buffer.append((u, v))
                valid_edges += 1

                if len(edge_buffer) >= chunk_size:
                    G.add_edges_from(edge_buffer)
                    edge_buffer.clear()
                    gc.collect()

                    print(
                        f"  -> 已读取行数: {total_lines:,}, "
                        f"有效边数约: {valid_edges:,}, "
                        f"当前节点数: {G.number_of_nodes():,}, "
                        f"当前边数: {G.number_of_edges():,}"
                    )

        if edge_buffer:
            G.add_edges_from(edge_buffer)
            edge_buffer.clear()
            gc.collect()

        print(f"  -> 图加载完成。节点数: {G.number_of_nodes():,}, 边数: {G.number_of_edges():,}")
        print(f"  -> 加载耗时: {time.time() - start_time:.2f} 秒")

        return G

    # ============================================================
    # 主循环：三张图逐张处理
    # ============================================================

    for input_graph_path in input_graph_paths:

        input_file = Path(input_graph_path)
        output_dir = input_file.parent

        print("\n" + "=" * 100)
        print(f"开始处理数据集: {input_file.name}")
        print("=" * 100)

        partition_cache_file = output_dir / f"{input_file.stem}-cache_partition.pkl"
        community_props_cache_file = output_dir / f"{input_file.stem}-cache_community_props.pkl"
        inter_community_cache_file = output_dir / f"{input_file.stem}-cache_inter_community.pkl"
        extensibility_cache_file = output_dir / f"{input_file.stem}-cache_extensibility.pkl"

        # ========================================================
        # [阶段 1] 加载图：替换原来的 nx.read_edgelist
        # ========================================================

        print(f"\n--- [主流程] 阶段 1/4: 正在从 '{input_file.name}' 加载图... ---")
        start_load_time = time.time()

        G = load_graph_light(input_graph_path)

        print(f"  -> 图加载完成。节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        print(f"  -> 耗时: {time.time() - start_load_time:.2f} 秒")

        gc.collect()

        # ========================================================
        # [阶段 2] 预计算
        # ========================================================

        print("\n--- [主流程] 阶段 2/4: 开始预计算所有属性，智能检查缓存... ---")

        # --------------------------------------------------------
        # [2a] 社群 partition 和 community_props
        # --------------------------------------------------------

        partition = None
        community_props = None

        if partition_cache_file.exists() and community_props_cache_file.exists():
            print(f"\n  [2a] 发现社群缓存，正在加载...")
            partition = safe_pickle_load(partition_cache_file, "partition")
            community_props = safe_pickle_load(community_props_cache_file, "community_props")

        if partition is None or community_props is None:
            print(f"\n  [2a] 未发现社群缓存或缓存损坏，正在重新计算社群...")
            start_time = time.time()

            partition, community_props = precompute_community_and_roles_accelerated(G)

            print(f"     - ✓ 社群计算完毕，耗时 {time.time() - start_time:.2f} 秒。")

            atomic_pickle_dump(partition, partition_cache_file)
            atomic_pickle_dump(community_props, community_props_cache_file)

            print("     - ✓ 社群结果已安全缓存到磁盘。")

        gc.collect()

        # --------------------------------------------------------
        # [2b] 社群间连接
        # --------------------------------------------------------

        inter_community_counts = None

        if inter_community_cache_file.exists():
            print(f"\n  [2b] 发现社群间连接缓存，正在加载...")
            inter_community_counts = safe_pickle_load(
                inter_community_cache_file,
                "inter_community_counts"
            )

        if inter_community_counts is None:
            print(f"\n  [2b] 未发现社群间连接缓存或缓存损坏，正在计算...")
            start_time = time.time()

            inter_community_counts = precompute_inter_community_edges(G, partition)

            print(f"     - ✓ 社群间连接计算完毕，耗时 {time.time() - start_time:.2f} 秒。")

            atomic_pickle_dump(inter_community_counts, inter_community_cache_file)

            print("     - ✓ 社群间连接结果已安全缓存到磁盘。")

        gc.collect()

        # --------------------------------------------------------
        # [2c] 延展性分数
        # --------------------------------------------------------

        extensibility_scores = None

        if extensibility_cache_file.exists():
            print(f"\n  [2c] 发现延展性分数缓存，正在加载...")
            extensibility_scores = safe_pickle_load(
                extensibility_cache_file,
                "extensibility_scores"
            )

        if extensibility_scores is None:
            print(f"\n  [2c] 未发现延展性分数缓存或缓存损坏，正在计算...")
            start_time = time.time()

            extensibility_scores = calculate_custom_extensibility_FINAL(G)

            print(f"     - ✓ 延展性分数计算完毕，耗时 {time.time() - start_time:.2f} 秒。")

            atomic_pickle_dump(extensibility_scores, extensibility_cache_file)

            print("     - ✓ 延展性分数结果已安全缓存到磁盘。")

        gc.collect()

        print("\n--- ✓ 所有预计算完成！ ---")

        # ========================================================
        # [阶段 3] 执行所有攻击策略 / Domirank
        # ========================================================

        process_all_strategies_in_one_pass(
            G=G,
            output_dir=input_file.parent,
            file_stem=input_file.stem,
            extensibility_scores=extensibility_scores,
            partition=partition,
            community_props=community_props,
            inter_community_counts=inter_community_counts
        )

        # ========================================================
        # [阶段 4] 释放内存
        # ========================================================

        print("\n--- [主流程] 阶段 4/4: 正在释放当前数据集内存... ---")

        del G
        del extensibility_scores
        del partition
        del community_props
        del inter_community_counts

        gc.collect()

        print(f"--- ✓ 数据集 {input_file.name} 处理完成，内存已释放。 ---")

    print("\n--- ✓ 所有数据集处理完成。程序执行完毕。 ---")

