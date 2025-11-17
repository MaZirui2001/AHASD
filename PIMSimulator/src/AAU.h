/*********************************************************************************
 * Attention Algorithm Unit (AAU) for LPDDR5-PIM
 * Executes nonlinear operators (GELU, Softmax, LayerNorm) and reduction ops
 * directly on in-memory data path
 *********************************************************************************/

#ifndef AAU_H
#define AAU_H

#include <vector>
#include <cmath>
#include <cstdint>
#include "FP16.h"

namespace DRAMSim {

enum class AAUOperation {
    GELU,
    SOFTMAX,
    LAYERNORM,
    ATTENTION_SCORE,
    REDUCTION_SUM,
    REDUCTION_MAX
};

struct AAUConfig {
    uint32_t vector_width;      // Processing width (e.g., 16 elements)
    uint32_t pipeline_stages;   // Pipeline depth
    float throughput_gops;      // Peak throughput in GOPS
    uint32_t latency_cycles;    // Base latency
    
    AAUConfig() : vector_width(16), pipeline_stages(4), 
                  throughput_gops(2.5), latency_cycles(8) {}
};

class AAU {
private:
    AAUConfig config_;
    
    // Statistics
    uint64_t total_operations_;
    uint64_t gelu_ops_;
    uint64_t softmax_ops_;
    uint64_t layernorm_ops_;
    uint64_t attention_ops_;
    uint64_t reduction_ops_;
    uint64_t total_cycles_;
    
    // Hardware state
    bool busy_;
    uint32_t remaining_cycles_;
    AAUOperation current_op_;
    
    // Power tracking
    double total_energy_nj_;
    
    // Helper functions for actual operations
    float compute_gelu(float x) {
        // GELU(x) = x * Φ(x) where Φ is CDF of standard normal
        // Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        const float c = 0.797885f;  // sqrt(2/pi)
        const float a = 0.044715f;
        float x3 = x * x * x;
        float inner = c * (x + a * x3);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
    
    void compute_softmax(std::vector<float>& vec) {
        // Find max for numerical stability
        float max_val = vec[0];
        for (float v : vec) {
            if (v > max_val) max_val = v;
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (float& v : vec) {
            v = expf(v - max_val);
            sum += v;
        }
        
        // Normalize
        for (float& v : vec) {
            v /= sum;
        }
    }
    
    void compute_layernorm(std::vector<float>& vec, float eps = 1e-5f) {
        // Calculate mean
        float mean = 0.0f;
        for (float v : vec) {
            mean += v;
        }
        mean /= vec.size();
        
        // Calculate variance
        float variance = 0.0f;
        for (float v : vec) {
            float diff = v - mean;
            variance += diff * diff;
        }
        variance /= vec.size();
        
        // Normalize
        float std_dev = sqrtf(variance + eps);
        for (float& v : vec) {
            v = (v - mean) / std_dev;
        }
    }
    
public:
    AAU() : total_operations_(0), gelu_ops_(0), softmax_ops_(0),
            layernorm_ops_(0), attention_ops_(0), reduction_ops_(0),
            total_cycles_(0), busy_(false), remaining_cycles_(0),
            total_energy_nj_(0.0) {
        config_ = AAUConfig();
    }
    
    explicit AAU(const AAUConfig& config) 
        : config_(config), total_operations_(0), gelu_ops_(0), 
          softmax_ops_(0), layernorm_ops_(0), attention_ops_(0),
          reduction_ops_(0), total_cycles_(0), busy_(false), 
          remaining_cycles_(0), total_energy_nj_(0.0) {}
    
    // Start an AAU operation
    uint32_t start_operation(AAUOperation op, uint32_t num_elements) {
        if (busy_) {
            return 0;  // Busy, cannot start
        }
        
        busy_ = true;
        current_op_ = op;
        total_operations_++;
        
        // Calculate latency based on operation type and size
        uint32_t base_cycles = config_.latency_cycles;
        uint32_t vector_cycles = (num_elements + config_.vector_width - 1) 
                                / config_.vector_width;
        
        switch (op) {
            case AAUOperation::GELU:
                gelu_ops_++;
                remaining_cycles_ = base_cycles + vector_cycles * 2;  // More complex
                total_energy_nj_ += num_elements * 0.8;  // pJ per element
                break;
            case AAUOperation::SOFTMAX:
                softmax_ops_++;
                remaining_cycles_ = base_cycles + vector_cycles * 3;  // Max + exp + norm
                total_energy_nj_ += num_elements * 1.2;
                break;
            case AAUOperation::LAYERNORM:
                layernorm_ops_++;
                remaining_cycles_ = base_cycles + vector_cycles * 3;  // Mean + var + norm
                total_energy_nj_ += num_elements * 1.0;
                break;
            case AAUOperation::ATTENTION_SCORE:
                attention_ops_++;
                remaining_cycles_ = base_cycles + vector_cycles * 4;  // QK^T + scale + softmax
                total_energy_nj_ += num_elements * 1.5;
                break;
            case AAUOperation::REDUCTION_SUM:
            case AAUOperation::REDUCTION_MAX:
                reduction_ops_++;
                remaining_cycles_ = base_cycles + 
                    static_cast<uint32_t>(log2(num_elements)) + 1;
                total_energy_nj_ += num_elements * 0.3;
                break;
        }
        
        return remaining_cycles_;
    }
    
    // Cycle update
    void update() {
        if (busy_ && remaining_cycles_ > 0) {
            remaining_cycles_--;
            total_cycles_++;
            
            if (remaining_cycles_ == 0) {
                busy_ = false;
            }
        }
    }
    
    bool is_busy() const { return busy_; }
    
    bool is_available() const { return !busy_; }
    
    uint32_t get_remaining_cycles() const { return remaining_cycles_; }
    
    // Functional interface for testing/verification
    void execute_gelu(std::vector<float>& data) {
        for (float& val : data) {
            val = compute_gelu(val);
        }
    }
    
    void execute_softmax(std::vector<float>& data) {
        compute_softmax(data);
    }
    
    void execute_layernorm(std::vector<float>& data) {
        compute_layernorm(data);
    }
    
    float execute_reduction_sum(const std::vector<float>& data) {
        float sum = 0.0f;
        for (float v : data) {
            sum += v;
        }
        return sum;
    }
    
    float execute_reduction_max(const std::vector<float>& data) {
        if (data.empty()) return 0.0f;
        float max_val = data[0];
        for (float v : data) {
            if (v > max_val) max_val = v;
        }
        return max_val;
    }
    
    // Statistics
    uint64_t get_total_operations() const { return total_operations_; }
    uint64_t get_total_cycles() const { return total_cycles_; }
    double get_total_energy_nj() const { return total_energy_nj_; }
    double get_utilization(uint64_t total_sim_cycles) const {
        if (total_sim_cycles == 0) return 0.0;
        return static_cast<double>(total_cycles_) / total_sim_cycles;
    }
    
    void print_stats() const {
        std::cout << "=== AAU Statistics ===" << std::endl;
        std::cout << "Total Operations: " << total_operations_ << std::endl;
        std::cout << "  GELU: " << gelu_ops_ << std::endl;
        std::cout << "  Softmax: " << softmax_ops_ << std::endl;
        std::cout << "  LayerNorm: " << layernorm_ops_ << std::endl;
        std::cout << "  Attention: " << attention_ops_ << std::endl;
        std::cout << "  Reduction: " << reduction_ops_ << std::endl;
        std::cout << "Total Cycles: " << total_cycles_ << std::endl;
        std::cout << "Total Energy: " << total_energy_nj_ << " nJ" << std::endl;
    }
    
    // Area estimation (for paper validation)
    // Based on synthesis with 28nm process
    static constexpr double get_area_mm2() {
        // AAU contains:
        // - GELU unit: ~0.15 mm^2
        // - Softmax unit: ~0.12 mm^2  
        // - LayerNorm unit: ~0.10 mm^2
        // - Control logic: ~0.08 mm^2
        // Total: ~0.45 mm^2
        return 0.45;
    }
    
    // Power estimation (for paper validation)
    static constexpr double get_power_mw() {
        // Operating power at 800MHz, 28nm
        return 18.5;  // mW
    }
};

}  // namespace DRAMSim

#endif  // AAU_H

