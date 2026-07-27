// =========================================================================
// 文件名: calculator_parallel_ULTRA_SLOW.cpp
// 描述: (终极慢速对照组-并行版) 在保留并行结构但不改变核心算法逻辑
//       的前提下，通过引入伪共享暗示、锁竞争、内存拷贝等手段，
//       最大化并行环境下的运行时间。
// =========================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
// #include <atomic>   // 【负优化】移除atomic，使用性能更差的锁
#include <iomanip>
#include <omp.h>
#include <mutex>       // 【负优化】引入互斥锁

// --- 进度条类 (完整定义) ---
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

// --- 使用极其低效的双重循环来计算交集 (逻辑不变) ---
int count_intersection_slowly(const std::vector<int>& v1, const std::vector<int>& v2) {
    int count = 0;
    for (int neighbor1 : v1) {
        for (int neighbor2 : v2) {
            if (neighbor1 == neighbor2) {
                count++;
            }
        }
    }
    return count;
}

inline double combinations_2(long long n) {
    if (n < 2) return 0.0;
    return static_cast<double>(n) * (n - 1) / 2.0;
}

// --- 核心计算逻辑 (逻辑不变) ---
double calculate_single_score_ultraslow_parallel(int u_local, const std::vector<std::vector<int>>& adj_cache);

// --- 主驱动函数 (已修改为“负优化”并行) ---
std::map<int, double> calculate_scores_in_memory(const std::string& adj_filepath, const std::string& log_filename) {
    std::cerr << "[终极慢速-并行] 正在加载图文件: " << adj_filepath << std::endl;
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
    
    // 保持逻辑不变：不进行排序和去重

    auto load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = load_end - load_start;
    std::cerr << "[终极慢速-并行] 图加载完成 (" << adj_list.size() << " 个节点), 耗时: " << load_duration.count() << " 秒。" << std::endl;

    std::cerr << "[终极慢速-并行] 正在使用低效交集法(并行)计算分数..." << std::endl;
    auto calc_start = std::chrono::high_resolution_clock::now();
    std::vector<double> scores(adj_list.size());
    size_t num_nodes = adj_list.size();
    ProgressBar progress_bar(num_nodes);
    
    // 【负优化】使用性能最差的锁来替代 atomic
    size_t progress_counter = 0;
    std::mutex progress_mutex;

    #pragma omp parallel for schedule(static, 1) // 【负优化】使用最差的调度策略
    for (size_t i = 0; i < num_nodes; ++i) {
        scores[i] = calculate_single_score_ultraslow_parallel(i, adj_list);
        
        // 【负优化】在粗粒度的锁中进行进度条更新，制造锁竞争
        size_t update_threshold = std::max((size_t)1, num_nodes / 1000); 
        
        std::lock_guard<std::mutex> lock(progress_mutex);
        progress_counter++;
        if (progress_counter % update_threshold == 0 || progress_counter == num_nodes) {
            progress_bar.update(progress_counter);
        }
    }
    
    auto calc_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> calc_duration = calc_end - calc_start;
    std::cerr << "[终极慢速-并行] 分数计算完成, 纯计算耗时: " << calc_duration.count() << " 秒。" << std::endl;

    std::ofstream log_file(log_filename);
    if (log_file.is_open()) {
        log_file << "Graph File: " << adj_filepath << "\n";
        log_file << "Method: Ultra Slow Pair (Parallel)\n";
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
        std::map<int, double> scores = calculate_scores_in_memory(argv[1], "performance_ultraslow_parallel.log");
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

double calculate_single_score_ultraslow_parallel(int u_local, const std::vector<std::vector<int>>& adj_cache) {
    if (u_local >= adj_cache.size()) return 0.0;
    const auto& neighbors_of_u = adj_cache[u_local];
    size_t num_neighbors = neighbors_of_u.size();
    if (num_neighbors < 2) return static_cast<double>(num_neighbors);

    // 【负优化】提高阈值，强制走完整计算
    const size_t DEGREE_THRESHOLD = 1000000;
    const int NUM_PAIR_SAMPLES = 3000;
    double total_overlap = 0.0;

    if (num_neighbors <= DEGREE_THRESHOLD) {
        for (size_t i = 0; i < num_neighbors; ++i) {
            for (size_t j = i + 1; j < num_neighbors; ++j) {
                int v1 = neighbors_of_u[i];
                int v2 = neighbors_of_u[j];
                
                if (v1 < 0 || v1 >= adj_cache.size() || v2 < 0 || v2 >= adj_cache.size()) continue;

                const double v_deg = static_cast<double>(adj_cache[v1].size());
                const double u_deg = static_cast<double>(adj_cache[v2].size());

                if (u_deg > 0 && ((std::abs(v_deg / u_deg) <= 0.1) || (std::abs(v_deg / u_deg) >= 10.0))) {
                    if(u_deg >= v_deg) {
                        total_overlap += v_deg;
                    } else {
                        total_overlap += u_deg;
                    }
                } else {
                    // 【负优化】增加不必要的内存拷贝
                    std::vector<int> v1_copy = adj_cache[v1];
                    std::vector<int> v2_copy = adj_cache[v2];
                    total_overlap += count_intersection_slowly(v1_copy, v2_copy);
                }
            }
        }
    } else {
        double sampled_overlap_sum = 0;
        int valid_samples = 0;
        
        // 【负优化】使用一个全局的、被锁保护的随机数生成器，制造序列化瓶颈
        static std::mt19937 shared_gen(std::random_device{}());
        static std::mutex gen_mutex;

        for (int k = 0; k < NUM_PAIR_SAMPLES; ++k) {
            size_t idx1, idx2;
            {
                std::lock_guard<std::mutex> lock(gen_mutex);
                std::uniform_int_distribution<size_t> dist(0, num_neighbors - 1);
                idx1 = dist(shared_gen);
                idx2 = dist(shared_gen);
            }

            if (idx1 == idx2) { k--; continue; }
            int v1 = neighbors_of_u[idx1];
            int v2 = neighbors_of_u[idx2];
            
            if (v1 >= 0 && v1 < adj_cache.size() && v2 >= 0 && v2 < adj_cache.size()) {
                // 【负优化】同样增加内存拷贝
                std::vector<int> v1_copy = adj_cache[v1];
                std::vector<int> v2_copy = adj_cache[v2];
                sampled_overlap_sum += count_intersection_slowly(v1_copy, v2_copy);
                valid_samples++;
            }
        }
        if (valid_samples > 0) {
            total_overlap = (sampled_overlap_sum / valid_samples) * combinations_2(num_neighbors);
        }
    }

    double num_pairs = combinations_2(num_neighbors);
    if (num_pairs == 0) return static_cast<double>(num_neighbors);
    double cohesion = total_overlap / num_pairs;
    return static_cast<double>(num_neighbors) / (1.0 + cohesion);
}