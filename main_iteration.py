import gc
import time
import domirank_weight as dr
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# 图形设置
A = 6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 24})

analytical = False
np.random.seed(42)

import numpy as np
import os


def create_robustness_plot(curves_dict, title, color_palette, output_filename, unweighted_data=None):
    """
    【修正版】绘制综合对比图，并在下方附带一个效率对比表格。
    - 能够健壮地处理列表或Numpy数组形式的输入数据。
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # --- 样式定义 ---
    linestyles = {
        'DomiRank': '-', 'Degree': '--', 'PageRank': ':',
        'Betweenness': '-.', 'default': (0, (5, 10))
    }
    
    # --- 颜色分配 ---
    cmap = plt.colormaps.get_cmap(color_palette)
    all_strategies = list(curves_dict.keys()) + (list(unweighted_data.keys()) if unweighted_data else [])
    num_strategies = len(all_strategies)
    colors = cmap(np.linspace(0, 1, num_strategies)) if num_strategies > 1 else [cmap(0.5)]

    # --- 绘制主数据 (curves_dict) ---
    for i, (strategy, results) in enumerate(curves_dict.items()):
        base_name = strategy.split(' (')[0]
        
        # 【【【关键修正 #1】】】: 在使用前将列表转换为Numpy数组
        removed_ratios_np = np.array(results['removed_ratios'])
        components_np = np.array(results['components'])#components,links
        
        ax.plot(
            removed_ratios_np * 100,
            components_np * 100,
            color=colors[i],
            linestyle=linestyles.get(base_name, linestyles['default']),
            linewidth=2, label=strategy, alpha=0.9
        )

    # --- 绘制次要数据 (unweighted_data) ---
    if unweighted_data:
        offset = len(curves_dict)
        for j, (strategy, results) in enumerate(unweighted_data.items()):
            base_name = strategy.split(' (')[0]

            # 【【【关键修正 #2】】】: 同样进行转换
            removed_ratios_np = np.array(results['removed_ratios'])
            components_np = np.array(results['components'])
            
            ax.plot(
                removed_ratios_np * 100,
                components_np * 100,
                color=colors[offset + j],
                linestyle='--', linewidth=2, label=strategy, alpha=0.9
            )

    # --- 图表装饰 ---
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Percentage of Nodes Removed (%)', fontsize=14)
    ax.set_ylabel('Relative Total Flow (%)', fontsize=14) # 注意Y轴标签，可以改为 'Relative LCC Size (%)'
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    
    # --- 图例 ---
    ax.legend(loc='upper right', fontsize=10)
    target_removal_ratio = 0.50

    # 1. 收集所有策略的数据
    all_curves_data = {}
    all_curves_data.update(curves_dict)
    if unweighted_data:
        all_curves_data.update(unweighted_data)

    # 2. 提取在目标点的数值
    flow_at_target = {}
    for strategy, results in all_curves_data.items():
        # 【【【关键修正 #3】】】: 确保用于计算的也是Numpy数组
        removed_ratios_np = np.array(results['removed_ratios'])
        components_np = np.array(results['components'])
        
        # 找到移除比例最接近目标点(50%)的那个点的索引
        idx_target = np.argmin(np.abs(removed_ratios_np - target_removal_ratio))
        
        # 获取该点的组件(component)值
        flow_value = components_np[idx_target]
        flow_at_target[strategy] = flow_value

    # --- 3. 创建并美化表格 ---
    if flow_at_target:
        sorted_strategies = sorted(flow_at_target, key=flow_at_target.get)
        
        table_data = [[f'{flow_at_target[strategy]*100:.2f}%'] for strategy in sorted_strategies]
        row_labels = [s for s in sorted_strategies]
        
        # 【【【Y轴标签修正建议】】】
        # 表格的标签应与Y轴数据一致。既然您绘制的是'components'，标签也应该是'Remaining LCC'
        col_label = f'Remaining LCC at {target_removal_ratio*100}% Removal'
        
        the_table = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=[col_label],
            loc='bottom',
            cellLoc='center',
            bbox=[0.0, -0.5, 1, 0.3]
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        
        for (i, j), cell in the_table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.subplots_adjust(bottom=0.4)

    # --- 保存 ---
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    # 确保文件夹存在
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

from scipy.sparse import csr_matrix
import time
import gc
from tqdm import tqdm

class FastGraphLoader:
    def __init__(self):
        self.mapping = {}
        self.reverse_mapping = [] # 使用 list 作为反向映射，更紧凑
        self.next_id = 0

    def _get_or_create_id(self, node_str):
        """在线获取或创建节点ID"""
        if node_str not in self.mapping:
            self.mapping[node_str] = self.next_id
            self.reverse_mapping.append(node_str)
            self.next_id += 1
        return self.mapping[node_str]

    def load_unified(self, primary_path, secondary_paths=[], is_weighted=True):
        """
        一次遍历完成映射和构建。
        primary_path: 主要的、包含权重的文件，将用它来构建主矩阵。
        secondary_paths: 其他文件，仅用于确保它们的节点也被包含在映射中。
        """
        print("--- ✓ 正在以“单次遍历，即时映射”模式高效加载图... ---")
        start_time = time.time()
        
        # --- 步骤 1: 扫描次要文件，仅用于补充映射 (轻量级) ---
        # 这一步是为了确保映射包含所有文件中的节点
        # 比如无权图中有一些孤立节点，也需要被映射
        print("  -> [步骤 1/3] 扫描次要文件以预热映射...")
        for path in secondary_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"预扫描 @ {path.split('/')[-1]}", unit="行"):
                    if line.startswith('#') or line.strip() == '': continue
                    parts = line.split()
                    if len(parts) >= 2:
                        self._get_or_create_id(parts[0])
                        self._get_or_create_id(parts[1])

        # --- 步骤 2: 主遍历，同时构建矩阵和最终映射 ---
        print(f"  -> [步骤 2/3] 主遍历文件 '{primary_path.split('/')[-1]}' 并构建矩阵...")
        
        # 预估边数以预分配 NumPy 数组，这是巨大的性能提升
        # 简单地通过文件大小估算行数
        try:
            import os
            num_lines = os.path.getsize(primary_path) // 40 # 估算，假设平均每行40字节
            estimated_edges = num_lines * 2 # 无向图
            print(f"    -> 预估边数: {estimated_edges}，将预分配内存...")
            rows = np.empty(estimated_edges, dtype=np.int32)
            cols = np.empty(estimated_edges, dtype=np.int32)
            data = np.empty(estimated_edges, dtype=np.float32)
        except:
            print("    -> 无法估算大小，将使用 Python 列表 (速度稍慢)。")
            rows, cols, data = [], [], [] # Fallback
            
        edge_count = 0
        with open(primary_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"构建矩阵 @ {primary_path.split('/')[-1]}", unit="行"):
                if line.startswith('#') or line.strip() == '': continue
                parts = line.split()
                if len(parts) < 2: continue
                
                u_idx = self._get_or_create_id(parts[0])
                v_idx = self._get_or_create_id(parts[1])

                weight = float(parts[2]) if is_weighted and len(parts) > 2 else 1.0
                
                # 直接填充 NumPy 数组，或 append 到列表
                if isinstance(rows, np.ndarray):
                    # 检查是否需要扩容
                    if edge_count + 2 > len(rows):
                        new_size = int(len(rows) * 1.5) # 扩容50%
                        rows = np.resize(rows, new_size)
                        cols = np.resize(cols, new_size)
                        data = np.resize(data, new_size)

                    rows[edge_count] = u_idx
                    cols[edge_count] = v_idx
                    data[edge_count] = weight
                    edge_count += 1
                    
                    rows[edge_count] = v_idx
                    cols[edge_count] = u_idx
                    data[edge_count] = weight
                    edge_count += 1
                else:
                    rows.append(u_idx); cols.append(v_idx); data.append(weight)
                    rows.append(v_idx); cols.append(u_idx); data.append(weight)

        # --- 步骤 3: 从填充好的数据构建矩阵 ---
        print("  -> [步骤 3/3] 从填充数据构建最终稀疏矩阵...")
        N = self.next_id
        
        if isinstance(rows, np.ndarray):
            # 如果使用 NumPy，裁剪掉未使用的部分
            final_rows = rows[:edge_count]
            final_cols = cols[:edge_count]
            final_data = data[:edge_count]
        else:
            final_rows, final_cols, final_data = rows, cols, data

        G_matrix = csr_matrix((final_data, (final_rows, final_cols)), shape=(N, N))
        
        print(f"\n--- ✓ 统一加载完成！耗时: {time.time() - start_time:.2f} 秒 ---")
        return G_matrix, self.mapping, {i: node for i, node in enumerate(self.reverse_mapping)}


# --- 如何使用 ---
if __name__ == '__main__':
    unweighted_path = "/home/wcy/domirank/domirank/Data/road-1500000/road.txt"
    weighted_path = "/home/wcy/domirank/domirank/Data/road-1500000/road-e-shengshe.txt"

    # --- 新的、统一的加载方式 ---
    loader = FastGraphLoader()
    GAdj_w, mapping, reverse_mapping = loader.load_unified(
        primary_path=weighted_path, 
        secondary_paths=[unweighted_path], 
        is_weighted=True
    )
    N = len(mapping)
    GAdj_u_data = np.ones(GAdj_w.nnz, dtype=np.float32)
    GAdj_u = csr_matrix((GAdj_u_data, GAdj_w.indices, GAdj_w.indptr), shape=(N, N))

    gc.collect()
    print(f"\n网络节点数: {N}")
    print(f"无权邻接矩阵 (shape: {GAdj_u.shape}, non-zeros: {GAdj_u.nnz})")
    print(f"加权邻接矩阵 (shape: {GAdj_w.shape}, non-zeros: {GAdj_w.nnz})")
    # --- 2. 生成攻击序列 ---
    print("\n[步骤 2] 正在生成四种攻击序列...")
    attack_sequences = {}
    start_time_total_calc = time.time()

    # # DomiRank (Weighted)
    print("  - 计算 1/4: DomiRank (on Weighted Graph)...")
    start_time = time.time()
    lambN_w = -2.805551052093506#dr.find_eigenvalue(GAdj_w, maxIter=500,dt=0.01, checkStep=10)
    print("la = ",lambN_w)
    sigma_w ,_ = dr.optimal_sigma(GAdj_w, analytical=analytical,iterationNo=10,endVal=lambN_w,dt = 0.1)
    print("sigma_w:",sigma_w)
    _, dist_domi_w = dr.domirank(GAdj_w, analytical=analytical, sigma=sigma_w,dt = 0.1,output_filename="frozen_nodes_dip.txt")
    attack_sequences['DomiRank (Weighted)'] = dr.generate_attack(dist_domi_w)
    print(f"    >>> 耗时: {time.time() - start_time:.2f}s")
    
    # DomiRank (Unweighted)
    print("  - 计算 4/4: DomiRank (on Unweighted Graph)...")
    start_time = time.time()
    lambN_u = dr.find_eigenvalue(GAdj_u, maxIter=500, dt=0.1,checkStep=10)
    print("la = ",lambN_u)
    sigma_u, _ = dr.optimal_sigma(GAdj_u, analytical=analytical,iterationNo=10,endVal=lambN_u,dt = 0.1)
    print("sigma_u:",sigma_u)
    _, dist_domi_u = dr.domirank(GAdj_u, analytical=analytical, sigma=sigma_u,dt = 0.1)
    attack_sequences['DomiRank (Unweighted)'] = dr.generate_attack(dist_domi_u)
    print(f"    >>> 耗时: {time.time() - start_time:.2f}s")

    print("所有攻击序列生成完毕！")
    print(f"    >>> 总耗时: {time.time() - start_time_total_calc:.2f}s")

    # # --- 3. 攻击模拟 ---
    print("\n[步骤 3] 分别在加权图和无权图上执行攻击模拟...")
    robustness_curves_weighted = {}  # 存储加权策略在加权图上的结果
    robustness_curves_unweighted = {}  # 存储无权策略在无权图上的结果

    for name, sequence in attack_sequences.items():
        num_to_print = min(25, len(sequence))
        print("攻击序列：",name)
        for i in range(num_to_print):
            node_id = sequence[i]
            # 如果您还想打印分数，需要稍微修改一下
            print(f"{i+1}. Attack Node: {node_id}")
        if 'DomiRank' in name :
            if '(Weighted)' in name:
                # 加权策略在加权图上测试
                print(f"  - 模拟攻击 (Weighted): {name}...")
                result = dr.network_attack_sampled(GAdj_w, sequence)
                
                robustness_curves_weighted[name] = result
            else:
                # 无权策略在无权图上测试
                print(f"  - 模拟攻击 (Unweighted): {name}...")
                result = dr.network_attack_sampled(GAdj_u, sequence)
                robustness_curves_unweighted[name] = result
        # else :
        #     if '(Weighted)' in name:
        #         # pass
        #         # 加权策略在加权图上测试
        #         print(f"  - 模拟攻击 (Weighted): {name}...")
        #         result = dr.network_attack_sampled(G_w, sequence)
        #         # result['components'] = dr.simulate_attack(G_w, sequence)
        #         robustness_curves_weighted[name] = result
        #     else:
        #     # 无权策略在无权图上测试
        #         print(f"  - 模拟攻击 (Unweighted): {name}...")
        #         result = dr.network_attack_sampled(G_u, sequence)
        #         robustness_curves_unweighted[name] = result

    print("所有模拟完成！")
    
    # # 加权策略结果
    if robustness_curves_weighted:
        create_robustness_plot(
            curves_dict=robustness_curves_weighted,
            title='Attack Strategies on Weighted Graph',
            color_palette='plasma',
            output_filename='figs_standard/weighted_strategies_on_weighted.png'
        )
    
    # # 无权策略结果
    if robustness_curves_unweighted:
        create_robustness_plot(
            curves_dict=robustness_curves_unweighted,
            title='Un Attack Strategies on Weighted Graph',
            color_palette='viridis',
            output_filename='figs_standard/unweighted_strategies_on_unweighted.png'
        )

    if robustness_curves_weighted or robustness_curves_unweighted:
        create_robustness_plot(
            curves_dict=robustness_curves_weighted,
            title='Un Attack Strategies on Weighted Graph',
            color_palette='viridis',
            output_filename='figs_standard/all_strategies_on_unweighted.png',
            unweighted_data=robustness_curves_unweighted
        )