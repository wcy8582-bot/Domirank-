# DomiRank⁺: Effective and Efficient DomiRank in Large Graphs

本仓库为论文 **《Effective and Efficient DomiRank in Large Graphs》** 的官方代码实现。

This repository contains the official implementation of the paper *"Effective and Efficient DomiRank in Large Graphs"*.

---

## 仓库结构 | Repository Structure

```
.
├── domirank-main/            # DomiRank⁺ Python 实现（论文主体算法）
│   ├── main_iteration.py     # DomiRank-Iter：原始 DomiRank 迭代求解器（基线）
│   ├── main_iteration+.py    # DomiRank-Iter⁺：加速迭代求解器（节点压缩 + 惰性更新 + Anderson 加速）
│   ├── main_ana+.py          # DomiRank-Anal⁺：解析求解器（共轭梯度法，具严格收敛保证）
│   ├── domirank_weight.py    # 边脆弱性加权模块（Ω = Ω_N · Ω_C）
│   ├── domirank_improve.py   # 改进的脆弱性计算模块
│   ├── domirank_结构压缩.py   # 结构等价节点压缩模块
│   └── domiran_iteration.py  # 基础迭代计算模块
│
├── cpp-extensibility/        # 边脆弱性计算的 C++ 高性能实现（对应论文的可扩展性部分，即将上传）
│   ├── calculator_fast_pivot.cpp    # 枢纽算法：基于主元（pivot）累加的快速边脆弱性计算
│   ├── calculator_superslow.cpp     # 邻居对算法：朴素邻居对枚举实现（正确性对照基线）
│   ├── performance_fast_pivot.log   # 枢纽算法性能日志
│   └── performance_superslow.log    # 邻居对算法性能日志
│
├── generator-union.py        # 结果加权脚本：对网络/结果进行脆弱性加权合并
│
├── Data/                     # 示例实验数据（4 个代表性网络，边列表格式：源节点 目标节点 权重）
│   ├── email/email.txt       # Email 网络
│   ├── facebook/facebook.txt # Facebook 网络
│   ├── DIP/Dip.txt           # DIP 蛋白质交互网络
│   └── hert/hert.txt         # Hetrec 网络
│
├── requirements.txt          # Python 依赖清单
└── README.md
```

> 说明：仓库中**不包含**网络生成（generator）相关代码；`Data/` 仅保留上述 4 个代表性网络，完整实验数据（如 Pokec 百万节点网络）可联系作者获取。

## 环境依赖 | Requirements

- Python ≥ 3.8，依赖安装：

```bash
pip install -r requirements.txt
```

- C++ 部分需要支持 C++11 及以上标准的编译器（g++ / MSVC）。

## 快速开始 | Quick Start

每个入口脚本均可独立运行（安装依赖后）：

```bash
# DomiRank-Iter：原始迭代求解器（基线）
python domirank-main/main_iteration.py

# DomiRank-Iter⁺：加速迭代求解器（推荐用于大规模图）
python domirank-main/main_iteration+.py

# DomiRank-Anal⁺：解析求解器（需要可认证的数值精度时使用）
python domirank-main/main_ana+.py

# 结果加权
python generator-union.py
```

C++ 边脆弱性计算：

```bash
cd cpp-extensibility
g++ -O3 -std=c++11 calculator_fast_pivot.cpp -o calculator_fast_pivot
./calculator_fast_pivot
```

## 算法说明 | Algorithm Overview

- **DomiRank⁺(Ω)** 通过边脆弱性 Ω = Ω_N · Ω_C 对邻接矩阵加权：
  - **Ω_N（多层邻域脆弱性）**：刻画节点的局部连接冗余度（默认层数 i = 1，即 2 跳邻域）；
  - **Ω_C（边社区脆弱性）**：基于 Leiden 社区划分，识别社区间桥接与长程瓶颈等全局结构。
- **DomiRank-Anal⁺**：将脆弱性加权后的方程组用共轭梯度法求解；系统矩阵可证明为对称正定（引理 5.1），误差以单调收缩因子为上界（引理 5.2）。
- **DomiRank-Iter⁺**：通过结构等价节点压缩、惰性更新与 Anderson 加速，在大规模图上实现 1–2 个数量级的效率提升。
- **C++ 延展性实现**：`calculator_fast_pivot` 采用基于主元的累加策略计算边脆弱性，复杂度显著优于朴素邻居对枚举（`calculator_superslow`，用于正确性验证）。

## 引用 | Citation

如果本代码对您的研究有帮助，请引用我们的论文：

```
Effective and Efficient DomiRank in Large Graphs
Long Yuan, Congyi Wang, Zi Chen, Kongzhang Hao, Wenjie Zhang, Xuemin Lin
```
