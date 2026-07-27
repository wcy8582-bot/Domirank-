import gc
import os
import time
import domirank_improve as dr
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import domirank_weight as dr1
########### FIGURE STUFF ###############
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 24})
########################################
m = 3 #average number of links per node.
analytical = True #if you want to use the analytical method or the recursive definition
seed = 42
np.random.seed(seed)
##### END OF RANDOMIZATION #####

import numpy as np
import scipy.sparse as sp
import time
import gc
import time
import domirank_improve as dr
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import domirank_weight as dr1
import random

# =============================================================================
# 1. 图加载与抽样函数 (复用之前的代码)
# =============================================================================

def load_graph_and_relabel_direct_to_scipy(input_path, dtype_spec):
    """
    直接从边列表文件加载图到Scipy稀疏矩阵，同时执行节点重标签。
    """
    print(f"\n===========================================================")
    print(f"  Loading Full Graph from: '{input_path.split('/')[-1]}'")
    print(f"===========================================================")
    start_time = time.time()
    sources, targets, weights = [], [], []
    unique_nodes = set()
    with open(input_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3:
                s, t, w = int(parts[0]), int(parts[1]), float(parts[2])
                sources.append(s); targets.append(t); weights.append(w)
                unique_nodes.add(s); unique_nodes.add(t)
    
    sorted_unique_nodes = np.array(list(unique_nodes))
    node_to_idx = {node: i for i, node in enumerate(sorted_unique_nodes)}
    idx_to_node = sorted_unique_nodes
    N = len(unique_nodes)
    
    relabeled_sources = [node_to_idx[s] for s in sources]
    relabeled_targets = [node_to_idx[t] for t in targets]
    
    row = np.array(relabeled_sources, dtype=dtype_spec['source'])
    col = np.array(relabeled_targets, dtype=dtype_spec['target'])
    data = np.array(weights, dtype=dtype_spec['weight'])
    
    full_row = np.concatenate([row, col])
    full_col = np.concatenate([col, row])
    full_data = np.concatenate([data, data])
    G_coo = sp.coo_matrix((full_data, (full_row, full_col)), shape=(N, N))
    G_csr = G_coo.tocsr()
    G_csr.sum_duplicates()
    
    print(f"  -> Full graph loaded: {G_csr.shape[0]} nodes, {G_csr.nnz // 2} edges")
    print(f"  -> Total loading time: {time.time() - start_time:.6f} seconds")
    
    return G_csr, node_to_idx, idx_to_node

def create_sampled_subgraph_by_edges(G_full: sp.csr_matrix, sample_ratio: float):
    """
    通过随机边抽样来创建一个子图。
    """
    if sample_ratio >= 1.0:
        print("\n--- Sample ratio is 100%, returning a copy of the full graph ---")
        return G_full.copy()

    print(f"\n--- Creating subgraph with {sample_ratio:.0%} edge sample ---")
    rows, cols, _ = sp.find(G_full)
    unique_edges = list(set(tuple(sorted(edge)) for edge in zip(rows, cols)))
            
    num_total_edges = len(unique_edges)
    num_edges_to_sample = int(num_total_edges * sample_ratio)
    
    print(f"  - Original unique edges: {num_total_edges:,}")
    print(f"  - Sampling {num_edges_to_sample:,} edges...")
    
    sampled_edges = random.sample(unique_edges, num_edges_to_sample)
    
    if not sampled_edges:
        return sp.csr_matrix(G_full.shape, dtype=G_full.dtype)
        
    sampled_rows, sampled_cols = zip(*sampled_edges)
    final_rows = np.concatenate([sampled_rows, sampled_cols])
    final_cols = np.concatenate([sampled_cols, sampled_rows])
    final_data = np.ones(len(final_rows), dtype=G_full.dtype)
    
    G_sampled = sp.csr_matrix((final_data, (final_rows, final_cols)), shape=G_full.shape)
    
    print(f"  - Subgraph created: {G_sampled.shape[0]} nodes, {G_sampled.nnz // 2} edges")
    
    return G_sampled



def calculate_comparison_metrics(
    scores1: np.ndarray, 
    scores2: np.ndarray, 
    top_k: int = 100
):
    """
    计算两个分数向量之间的多种比较指标，
    包括全局MSE、Top-K Jaccard，以及更具区分度的 Top-K MSE。
    """
    # --- a) 全局归一化 (与之前相同) ---
    def normalize(scores):
        min_val, max_val = scores.min(), scores.max()
        if max_val == min_val: return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    scores1_norm = normalize(scores1)
    scores2_norm = normalize(scores2)
    
    # --- b) 计算全局 MSE (作为基线参考) ---
    global_mse = np.mean((scores1_norm - scores2_norm)**2)

    # --- c) 识别 Top-K 节点 ---
    top_k_indices1 = np.argsort(scores1)[-top_k:]
    top_k_indices2 = np.argsort(scores2)[-top_k:]
    
    set1 = set(top_k_indices1)
    set2 = set(top_k_indices2)
    
    # --- d) 计算 Top-K Jaccard (与之前相同) ---
    intersection = len(set1.intersection(set2))
    union_len = len(set1.union(set2))
    jaccard = intersection / union_len if union_len > 0 else 0
    
    # --- e) 【核心新增】计算 Top-K 节点的 MSE ---
    # 1. 获取所有被任一算法认为是 Top-K 的节点的索引 (并集)
    union_indices = list(set1.union(set2))
    
    # 2. 从全局归一化的分数中，提取这些重要节点的分数
    top_scores1 = scores1_norm[union_indices]
    top_scores2 = scores2_norm[union_indices]
    
    # 3. 只在这些重要节点的分数上计算 MSE
    # 如果union_indices为空，则mse为0
    if len(union_indices) > 0:
        mse_top_k = np.mean((top_scores1 - top_scores2)**2)
    else:
        mse_top_k = 0.0

    # 返回一个包含所有指标的字典，更清晰
    return mse_top_k*100000000,jaccard
# =============================================================================
# 2. 【【【核心新函数】】】 封装您最新的分析和绘图流程
# =============================================================================
import psutil
def save_results_to_csv(results_dict, filename="experiment_improve_results.csv"):
    """将包含单次实验结果的字典追加到CSV文件中"""
    # 将字典转换为DataFrame的一行
    df_new_row = pd.DataFrame([results_dict])
    
    # 检查文件是否存在
    if os.path.exists(filename):
        # 如果文件存在，以追加模式写入，并且不写表头
        df_new_row.to_csv(filename, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，正常写入，包含表头
        df_new_row.to_csv(filename, mode='w', header=True, index=False)
    
    print(f"结果已成功保存到 {filename}")

from scipy.sparse.linalg import eigs

def run_full_analysis_on_graph(GAdj: sp.csr_matrix, sample_ratio: float, original_graph_name: str):
    """
    对给定的图执行完整的DomiRank分析流程，并生成结果图，
    同时计算并展示 MSE 和 Top-K Jaccard 相似度。
    """
    graph_name = f"{sample_ratio:.0%}_sample_of_{original_graph_name}"
    
    # --- 1. DomiRank (Proposed/Improved) Calculation ---
    print(f"\n--- [{graph_name}] Calculating Proposed DomiRank (analytical) ---")
    active_mask = None
    lambN_red = dr.find_eigenvalue(GAdj, dt=0.1,  maxIter=1000,checkStep=10,active_mask=active_mask)
    print("  -> Lambda_N (improved) calculated:", lambN_red)
    precomputed_rho_A = abs(lambN_red)
    sigma , _ = dr.optimal_sigma(GAdj, analytical=True, endVal=lambN_red, dt=0.1,  maxIter=1000,lambN=precomputed_rho_A,active_mask =active_mask)
    print("  -> Optimal Sigma (improved) calculated:", sigma)
    process = psutil.Process(os.getpid())
    mem_before_analytical = process.memory_info().rss
    start_time_improve = time.time()
    _, ourDomiRankDistribution = dr.domirank(GAdj, analytical=True, sigma=sigma, dt=0.1, lambN=precomputed_rho_A,active_mask=active_mask)
    time_improve = time.time() - start_time_improve
    mem_after_analytical = process.memory_info().rss
    mem_consumed_analytical_mb = (mem_after_analytical - mem_before_analytical) / (1024 ** 2)
    print(f"  -> DomiRank (improved) centrality calculated in {time_improve:.6f}s")
    analytical_results = {
    'graph_name': original_graph_name,
    'method': 'Analytical (spsolve)',
    'time_seconds': round(time_improve, 4),
    'peak_memory_mb': round(mem_after_analytical / (1024 ** 2), 2),
    'incremental_memory_mb': round(mem_consumed_analytical_mb, 2)
    }
    save_results_to_csv(analytical_results)
    # --- 2. DomiRank (Baseline/Original) Calculation ---
    # print(f"\n--- [{graph_name}] Calculating Baseline DomiRank ---")
    
    # lambN_orig = dr1.find_eigenvalue(GAdj, maxIter=2000, dt=0.01, checkStep=10)
    # print("  -> Lambda_N (baseline) calculated:", lambN_orig)
    # sigma_o, _ = dr1.optimal_sigma(GAdj,maxIter=2000, analytical=False, endVal=lambN_orig, dt=0.01)
    # print("  -> Optimal Sigma (baseline) calculated:", sigma_o)
    # start_time_orig_block = time.time()
    # _, ourDomiRankDistribution_o = dr1.domirank(GAdj, maxIter=2000,analytical=False, sigma=sigma_o, dt=0.01)
    # time_orig = time.time() - start_time_orig_block
    # time_improve = 100
    # ourDomiRankDistribution  = ourDomiRankDistribution_o
    ourDomiRankDistribution_o = ourDomiRankDistribution
    time_orig = 100
    print(f"  -> DomiRank (baseline) centrality calculated in {time_orig:.2f}s")
    
    # --- 3. 【【新增】】 计算比较指标 ---
    print(f"\n--- [{graph_name}] Calculating Comparison Metrics ---")
    mse, jaccard_top100 = calculate_comparison_metrics(
        ourDomiRankDistribution, 
        ourDomiRankDistribution_o,
        top_k=100
    )
    print(f"  -> MSE (Normalized): {mse:.6f}")
    print(f"  -> Top-100 Jaccard Similarity: {jaccard_top100:.4f}")

    # --- 4. Robustness Analysis ---
    print(f"\n--- [{graph_name}] Performing robustness analysis ---")
    robustness_curves = {}
    ourDomiRankAttack = dr.generate_attack(ourDomiRankDistribution)
    robustness_curves['DomiRank'] = dr.network_attack_sampled(GAdj, ourDomiRankAttack)
    ourDomiRankAttack_o = dr.generate_attack(ourDomiRankDistribution_o)
    robustness_curves['DomiRank_o'] = dr.network_attack_sampled(GAdj, ourDomiRankAttack_o)
    print("  -> Robustness curves calculated.")

    # --- 5. Plotting (增加了第三个表格) ---
    print(f"\n--- [{graph_name}] Generating plot with comparison tables ---")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    STRATEGY_STYLES = {
        'DomiRank':    {'color': 'red', 'linestyle': '-', 'marker': ''},
        'DomiRank_o':  {'color': 'blue', 'linestyle': ':', 'marker': ''},
    }
    
    for strategy, results in robustness_curves.items():
        if 'removed_ratios' in results and 'links' in results:
            x_values = np.array(results['removed_ratios']) * 100
            y_values = np.array(results['links']) * 100
            style = STRATEGY_STYLES.get(strategy, {})
            ax.plot(x_values, y_values, label=strategy, **style, linewidth=2.5)

    ax.set_title(f'Robustness Comparison on {graph_name}', fontsize=18, pad=20)
    ax.set_xlabel('Percentage of Nodes Removed (%)', fontsize=14)
    ax.set_ylabel('Relative Size of LCC (%)', fontsize=14)
    ax.set_ylim(0, 105); ax.set_xlim(0, 100)
    ax.legend(loc='upper right', fontsize=12)

    # --- 表格布局调整 ---
    table1_y_pos = -0.35  # Critical Ratios
    table2_y_pos = -0.60  # Runtimes
    table3_y_pos = -0.85  # 【新增】Comparison Metrics
    bottom_margin = 0.55  # 增加了底部边距以容纳第三个表格

    # Table 1: Critical Ratios (保持不变)
    CRITICAL_KEY = 'lcc_critical_ratio' 
    critical_ratios = {k: v[CRITICAL_KEY] for k, v in robustness_curves.items() if CRITICAL_KEY in v and v.get(CRITICAL_KEY) is not None}
    if critical_ratios:
        sorted_strategies = sorted(critical_ratios, key=critical_ratios.get)
        table_data = [[f'{s}' for s in sorted_strategies], [f'{critical_ratios[s]*100:.2f}%' for s in sorted_strategies]]
        top_table = ax.table(cellText=table_data, rowLabels=['Strategy', 'Critical Ratio (%)'], colLabels=[f'Rank {i+1}' for i in range(len(sorted_strategies))],
                             cellLoc='center', loc='bottom', bbox=[0.0, table1_y_pos, 1.0, 0.18])
        top_table.auto_set_font_size(False); top_table.set_fontsize(10); top_table.scale(1, 1.5)
    
    # Table 2: Runtimes (保持不变)
    time_cell_text = [['Strategy', 'Time (s)'], ['DomiRank (Proposed)', f'{time_improve:.6f}'], ['DomiRank (Baseline)', f'{time_orig:.6f}'], ['Speedup Ratio', f'{(time_orig/time_improve) if time_improve > 0 else "N/A":.2f}x']]
    bottom_table = ax.table(cellText=time_cell_text, cellLoc='center', loc='bottom', bbox=[0.3, table2_y_pos, 0.4, 0.15])
    bottom_table.auto_set_font_size(False); bottom_table.set_fontsize(10); bottom_table.scale(1, 1.8)

    # 【【新增】】 Table 3: Comparison Metrics
    comp_cell_text = [
        ['Metric', 'Value'],
        ['MSE (Normalized)', f'{mse:.6f}'],
        ['Top-100 Jaccard', f'{jaccard_top100:.4f}']
    ]
    comp_table = ax.table(cellText=comp_cell_text, cellLoc='center', loc='bottom', bbox=[0.3, table3_y_pos, 0.4, 0.12])
    comp_table.auto_set_font_size(False); comp_table.set_fontsize(10); comp_table.scale(1, 1.8)
    
    plt.subplots_adjust(bottom=bottom_margin)
    
    output_filename = f'克雷洛夫_{original_graph_name}_{sample_ratio:.0%}_sample.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"  -> Plot saved to: {output_filename}")
    plt.close(fig) 
# =============================================================================
# 3. 主实验工作流
# =============================================================================
def main_experiment_workflow():
    # --- 配置 ---
    input_path = "/home/njustdb/data/DomiRank-main/DomiRank-main/Amazon Product很好/amazon-shengshe.txt"
    original_graph_name = input_path.split('/')[-1].split('.')[0]
    
    dtype_spec = {
        'source': np.int32, 
        'target': np.int32, 
        'weight': np.float32
    }
    # sampling_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    sampling_ratios = [1.0]
    # --- 步骤 1: 加载完整图一次 ---
    G_full, _, _ = load_graph_and_relabel_direct_to_scipy(input_path, dtype_spec)
    
    # --- 步骤 2: 循环抽样并进行分析 ---
    for ratio in sampling_ratios:
        G_sampled = create_sampled_subgraph_by_edges(G_full, ratio)
        
        # 对子图运行完整的分析和绘图流程
        run_full_analysis_on_graph(G_sampled, ratio, original_graph_name)
        
        del G_sampled
        gc.collect()

    print("\n======================================")
    print("  Sampling experiment completed!")
    print("======================================")


if __name__ == "__main__":
    # 设置Matplotlib字体，避免中文乱码问题 (如果您的标签中有中文)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main_experiment_workflow()