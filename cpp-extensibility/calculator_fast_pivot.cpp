#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <map>
#include <random>
#include <atomic> // 不再需要
#include <iomanip>
#include <omp.h>

// --- 进度条类 (无变化) ---
class ProgressBar {
public:
    ProgressBar(size_t total, int bar_width = 70) : total_ticks(total), width(bar_width) {}
    void update(size_t current_progress) {
        float percentage = static_cast<float>(current_progress) / total_ticks;
        int filled_width = static_cast<int>(width * percentage);
        std::cout << "\r[";
        for (int i = 0; i < width; ++i) std::cout << (i < filled_width ? "█" : "-");
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage * 100.0 << "%"
                  << " (" << current_progress << "/" << total_ticks << ")" << std::flush;
    }
    ~ProgressBar() { std::cout << std::endl; }
private:
    size_t total_ticks;
    int width;
};

// --- 辅助函数 (无变化) ---
int count_sorted_intersection(const std::vector<int>& v1, const std::vector<int>& v2) {
    int count = 0;
    size_t i = 0, j = 0;
    while (i < v1.size() && j < v2.size()) {
        if (v1[i] < v2[j]) i++;
        else if (v2[j] < v1[i]) j++;
        else { count++; i++; j++; }
    }
    return count;
}

inline double combinations_2(long long n) {
    if (n < 2) return 0.0;
    return static_cast<double>(n) * (n - 1) / 2.0;
}

// --- 核心计算逻辑 (无变化) ---
double execute_pivot_method(int u_local, const std::vector<std::vector<int>>& adj_cache);
double execute_pair_method(int u_local, const std::vector<std::vector<int>>& adj_cache);
double calculate_single_score_pivot_from_cache(int u_local, const std::vector<std::vector<int>>& adj_cache);

// --- 主驱动函数 (已修改为单线程) ---
std::map<int, double> calculate_scores_in_memory(const std::string& adj_filepath, const std::string& log_filename) {
    std::cerr << "[优化组-单线程] 正在加载图文件: " << adj_filepath << std::endl;
    auto load_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> adj_list;
    std::ifstream file(adj_filepath);
    if (!file.is_open()) throw std::runtime_error("无法打开图文件: " + adj_filepath);
    int max_node_id = -1;
    std::vector<std::pair<int, int>> edges;
    int u, v; double weight;
    while (file >> u >> v >> weight) {
        edges.push_back({u, v});
        if (u > max_node_id) max_node_id = u;
        if (v > max_node_id) max_node_id = v;
    }
    file.close();
    adj_list.resize(max_node_id + 1);
    for(const auto& edge : edges) {
        adj_list[edge.first].push_back(edge.second);
        adj_list[edge.second].push_back(edge.first);
    }

    // --- 【修改点】移除了图加载后的并行排序 ---
    for (size_t i = 0; i < adj_list.size(); ++i) {
        std::sort(adj_list[i].begin(), adj_list[i].end());
        adj_list[i].erase(std::unique(adj_list[i].begin(), adj_list[i].end()), adj_list[i].end());
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = load_end - load_start;
    std::cerr << "[优化组-单线程] 图加载完成 (" << adj_list.size() << " 个节点), 耗时: " << load_duration.count() << " 秒。" << std::endl;

    std::cerr << "[优化组-单线程] 正在使用枢纽法(单线程)计算分数..." << std::endl;
    auto calc_start = std::chrono::high_resolution_clock::now();
    std::vector<double> scores(adj_list.size());
    size_t num_nodes = adj_list.size();
    ProgressBar progress_bar(num_nodes);
    size_t update_threshold = std::max((size_t)1, num_nodes / 1000);

    // --- 【修改点】移除了主计算循环的并行指令 ---
    for (size_t i = 0; i < num_nodes; ++i) {
        scores[i] = calculate_single_score_pivot_from_cache(i, adj_list);
        
        // 进度条更新现在是自然的单线程，无需任何同步机制
        if ((i + 1) % update_threshold == 0 || (i + 1) == num_nodes) {
            progress_bar.update(i + 1);
        }
    }
    auto calc_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> calc_duration = calc_end - calc_start;
    std::cerr << "[优化组-单线程] 分数计算完成, 纯计算耗时: " << calc_duration.count() << " 秒。" << std::endl;

    // 日志记录部分无变化
    std::ofstream log_file(log_filename);
    if (log_file.is_open()) {
        log_file << "Graph File: " << adj_filepath << "\n";
        log_file << "Method: Fast Pivot (Single Thread)\n";
        log_file << "Loading Time (s): " << load_duration.count() << "\n";
        log_file << "Pure Calculation Time (s): " << calc_duration.count() << "\n";
        log_file.close();
        std::cerr << "[日志] 性能数据已保存到 " << log_filename << std::endl;
    } else {
        std::cerr << "[错误] 无法打开日志文件 " << log_filename << " 进行写入！" << std::endl;
    }

    std::map<int, double> result_map;
    for (size_t i = 0; i < scores.size(); ++i) result_map[static_cast<int>(i)] = scores[i];
    return result_map;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <adj_filepath>" << std::endl;
        return 1;
    }
    try {
        auto total_start = std::chrono::high_resolution_clock::now();
        std::map<int, double> scores = calculate_scores_in_memory(argv[1], "performance_fast_pivot_single_thread.log");
        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "\n--- 计算完成 ---" << std::endl;
        std::cout << "总耗时: " << std::chrono::duration<double>(total_end - total_start).count() << " 秒。" << std::endl;
        
        std::vector<std::pair<int, double>> sorted_scores;
        for(const auto& pair : scores) sorted_scores.push_back(pair);
        std::sort(sorted_scores.begin(), sorted_scores.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
        std::cout << "\n--- 分数最高的前10个节点 ---" << std::endl;
        for(size_t i = 0; i < std::min((size_t)10, sorted_scores.size()); ++i) {
            std::cout << "节点: " << sorted_scores[i].first << ", 分数: " << sorted_scores[i].second << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// --- 核心计算逻辑实现 (无变化) ---
double execute_pivot_method(int u_local, const std::vector<std::vector<int>>& adj_cache) {
    const double SYMMETRY_TOLERANCE = 0.5;
    const auto& neighbors_of_u = adj_cache[u_local];
    const double u_deg = static_cast<double>(neighbors_of_u.size());
    std::unordered_map<int, int> pivot_counts;
    int symmetric_neighbor_count = 0;
    for (int v_local : neighbors_of_u) {
        if (v_local < 0 || v_local >= adj_cache.size()) continue;
        const double v_deg = static_cast<double>(adj_cache[v_local].size());
        bool is_symmetric = false;
        if (v_deg > 8000 && u_deg > 8000) {
            if (u_deg > 0 && ((std::abs(v_deg - u_deg) / std::max(v_deg , u_deg) < SYMMETRY_TOLERANCE) ||(std::abs(v_deg / u_deg) <= 1.5)||(std::abs(v_deg / u_deg) >= 0.5)|| (std::abs(v_deg / u_deg) <= 0.1) || (std::abs(v_deg / u_deg) >= 10.0))) {
                symmetric_neighbor_count++;
                is_symmetric = true;
            }
        }
        if (!is_symmetric) {
            for (int w_local : adj_cache[v_local]) {
                if (w_local == u_local || w_local < 0 || w_local >= adj_cache.size()) continue;
                pivot_counts[w_local]++;
            }
        }
    }
    double total_overlap = 0.0;
    for (const auto& pair : pivot_counts) {
        if (pair.second >= 2) total_overlap += combinations_2(pair.second);
    }
    if (symmetric_neighbor_count >= 2) {
        total_overlap += combinations_2(symmetric_neighbor_count) * u_deg;
    }
    return total_overlap;
}

double execute_pair_method(int u_local, const std::vector<std::vector<int>>& adj_cache) {
    const auto& neighbors_of_u = adj_cache[u_local];
    double total_overlap = 0.0;
    for (size_t i = 0; i < neighbors_of_u.size(); ++i) {
        for (size_t j = i + 1; j < neighbors_of_u.size(); ++j) {
            int v1 = neighbors_of_u[i], v2 = neighbors_of_u[j];
            if (v1 >= 0 && v1 < adj_cache.size() && v2 >= 0 && v2 < adj_cache.size()) {
                total_overlap += count_sorted_intersection(adj_cache[v1], adj_cache[v2]);
            }
        }
    }
    return total_overlap;
}

double calculate_single_score_pivot_from_cache(int u_local, const std::vector<std::vector<int>>& adj_cache) {
    const auto& neighbors_of_u = adj_cache[u_local];
    size_t num_neighbors = neighbors_of_u.size();
    if (num_neighbors < 2) return static_cast<double>(num_neighbors);

    const size_t DEGREE_THRESHOLD = 10000;
    const int NUM_SAMPLES = 4000;
    double total_overlap = 0.0;

    if (num_neighbors <= DEGREE_THRESHOLD) {
        total_overlap = execute_pivot_method(u_local, adj_cache);
    }
    else{
        std::unordered_map<int, int> pivot_counts;
            int symmetric_neighbor_count = 0;
            
            // 【修改点】使用 static 随机数生成器，适用于单线程环境
            static std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, num_neighbors - 1);

            // --- 步骤 2: 进行 NUM_SAMPLES 次采样 ---
            for (int k = 0; k < NUM_SAMPLES; ++k) {
                // a. 随机选择一个 u 的邻居 v
                size_t v_idx = dist(gen);
                int v_local = neighbors_of_u[v_idx];
                
                if (v_local < 0 || v_local >= adj_cache.size()) continue;

                // b. 对被选中的 v 执行与 execute_pivot_method 相同的内部逻辑
                const double v_deg = static_cast<double>(adj_cache[v_local].size());
                const double u_deg = static_cast<double>(num_neighbors);
                const double SYMMETRY_TOLERANCE = 0.01;
                
                if (v_deg > 10000 && u_deg > 10000) {
                    if (u_deg > 0 && ((std::abs(v_deg - u_deg) / std::max(v_deg , u_deg) < SYMMETRY_TOLERANCE) ||(std::abs(v_deg / u_deg) <= 1.5)||(std::abs(v_deg / u_deg) >= 0.5)|| (std::abs(v_deg / u_deg) <= 0.1) || (std::abs(v_deg / u_deg) >= 10.0))) {
                        symmetric_neighbor_count++;
                    }
                } else {
                    for (int w_local : adj_cache[v_local]) {
                        if (w_local == u_local) continue;
                        if (w_local < 0 || w_local >= adj_cache.size()) continue;
                        pivot_counts[w_local]++;
                    }
                }
            }

            // --- 步骤 3: 从采样结果中计算“样本重叠度” ---
            double sampled_total_overlap = 0.0;
            for (const auto& pair : pivot_counts) {
                int common_count = pair.second;
                if (common_count >= 2) {
                    sampled_total_overlap += static_cast<double>(common_count) * (common_count - 1) / 2.0;
                }
            }
            if (symmetric_neighbor_count >= 2) {
                double symmetric_pairs = static_cast<double>(symmetric_neighbor_count) * (symmetric_neighbor_count - 1) / 2.0;
                sampled_total_overlap += symmetric_pairs * num_neighbors;
            }

            // --- 步骤 4: 将样本重叠度放大，估算出总重叠度 ---
            if (NUM_SAMPLES > 0) {
                double avg_overlap_per_sampled_v = sampled_total_overlap / NUM_SAMPLES;
                total_overlap = avg_overlap_per_sampled_v * num_neighbors;
            }
        }

        // --- 最终分数计算 (逻辑完全不变) ---
        double num_pairs = combinations_2(num_neighbors);
        if (num_pairs == 0) return static_cast<double>(num_neighbors);
        double cohesion = total_overlap / num_pairs;
        return static_cast<double>(num_neighbors) / (1.0 + cohesion);
}