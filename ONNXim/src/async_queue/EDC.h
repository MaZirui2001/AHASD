#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include "../Common.h"

// Entropy-History-Aware Drafting Control (EDC) Module
// Combines historical prediction entropy with leading draft batches
// to perform hardware-level online learning

namespace AHASD {

constexpr uint32_t LEHT_SIZE = 8;  // Local Entropy History Table size
constexpr uint32_t PHT_SIZE = 512; // Pattern History Table size (2^9)
constexpr uint32_t PHT_COUNTER_BITS = 2;  // 2-bit saturating counter
constexpr float H_MAX = 10.0f;  // Maximum entropy value

// 2-bit saturating counter states
enum class CounterState : uint8_t {
    STRONGLY_NOT_TAKEN = 0,
    WEAKLY_NOT_TAKEN = 1,
    WEAKLY_TAKEN = 2,
    STRONGLY_TAKEN = 3
};

class EDC {
private:
    // Local Entropy History Table (LEHT) - stores recent entropy buckets
    std::vector<uint8_t> leht_;
    
    // Local Commit Entropy History Table (LCEHT) - stores verified entropy
    std::vector<uint8_t> lceht_;
    
    // Leading Length Register (LLR) - 3-bit counter
    uint8_t llr_;
    
    // Pattern History Table (PHT) - 512 entries with 2-bit counters
    std::vector<CounterState> pht_;
    
    // Statistics
    uint64_t total_predictions_;
    uint64_t correct_predictions_;
    uint64_t suppressed_drafts_;
    uint64_t total_drafts_;
    
    // Configuration
    float h_max_;
    uint32_t leht_ptr_;  // Circular buffer pointer
    
    // Helper functions
    uint8_t entropy_to_bucket(float entropy) const {
        // Map entropy to one of 8 buckets [0,7]
        if (entropy < 0.0f) entropy = 0.0f;
        if (entropy > h_max_) entropy = h_max_;
        return static_cast<uint8_t>((entropy / h_max_) * 7.99f);
    }
    
    uint16_t calculate_pht_index() const {
        // Calculate PHT index from LEHT groups and LLR
        // Input_PHT = {avg(H_{4-7}), avg(H_{0-3}), LLR}
        
        // Group 1: H_{0-3}
        uint32_t sum_low = 0;
        for (uint32_t i = 0; i < 4; i++) {
            sum_low += leht_[i];
        }
        uint8_t avg_low = sum_low / 4;  // 3 bits
        
        // Group 2: H_{4-7}
        uint32_t sum_high = 0;
        for (uint32_t i = 4; i < 8; i++) {
            sum_high += leht_[i];
        }
        uint8_t avg_high = sum_high / 4;  // 3 bits
        
        // Concatenate: {avg_high[2:0], avg_low[2:0], llr[2:0]} = 9 bits
        uint16_t index = (avg_high << 6) | (avg_low << 3) | llr_;
        return index & 0x1FF;  // Ensure 9-bit
    }
    
    void update_counter(CounterState& counter, bool taken) {
        uint8_t val = static_cast<uint8_t>(counter);
        if (taken && val < 3) {
            counter = static_cast<CounterState>(val + 1);
        } else if (!taken && val > 0) {
            counter = static_cast<CounterState>(val - 1);
        }
    }

public:
    EDC() : llr_(0), h_max_(H_MAX), leht_ptr_(0),
            total_predictions_(0), correct_predictions_(0),
            suppressed_drafts_(0), total_drafts_(0) {
        leht_.resize(LEHT_SIZE, 0);
        lceht_.resize(LEHT_SIZE, 0);
        pht_.resize(PHT_SIZE, CounterState::WEAKLY_TAKEN);
    }
    
    // Called after each draft batch generation
    bool should_continue_drafting(float avg_entropy) {
        total_drafts_++;
        
        // Map entropy to bucket and update LEHT
        uint8_t bucket = entropy_to_bucket(avg_entropy);
        leht_[leht_ptr_] = bucket;
        leht_ptr_ = (leht_ptr_ + 1) % LEHT_SIZE;
        
        // Increment LLR (3-bit, saturates at 7)
        if (llr_ < 7) {
            llr_++;
        }
        
        // Calculate PHT index and make prediction
        uint16_t pht_index = calculate_pht_index();
        CounterState prediction = pht_[pht_index];
        
        total_predictions_++;
        
        // MSB of counter determines prediction
        bool should_continue = (static_cast<uint8_t>(prediction) >= 2);
        
        if (!should_continue) {
            suppressed_drafts_++;
        }
        
        return should_continue;
    }
    
    // Called after NPU verification completes
    void update_on_verification(bool fully_accepted, uint32_t accepted_count) {
        // Decrement LLR
        if (llr_ > 0) {
            llr_--;
        }
        
        // If accepted, commit LEHT to LCEHT
        if (fully_accepted) {
            lceht_ = leht_;
            correct_predictions_++;
        } else {
            // Rollback: restore LEHT from LCEHT
            leht_ = lceht_;
        }
        
        // Update PHT based on verification result
        uint16_t pht_index = calculate_pht_index();
        update_counter(pht_[pht_index], fully_accepted);
    }
    
    // Reset state (for new inference sequence)
    void reset() {
        std::fill(leht_.begin(), leht_.end(), 0);
        std::fill(lceht_.begin(), lceht_.end(), 0);
        llr_ = 0;
        leht_ptr_ = 0;
    }
    
    // Getters for current state
    uint8_t get_llr() const { return llr_; }
    
    const std::vector<uint8_t>& get_leht() const { return leht_; }
    
    // Statistics
    double get_prediction_accuracy() const {
        if (total_predictions_ == 0) return 0.0;
        return static_cast<double>(correct_predictions_) / total_predictions_;
    }
    
    double get_suppression_rate() const {
        if (total_drafts_ == 0) return 0.0;
        return static_cast<double>(suppressed_drafts_) / total_drafts_;
    }
    
    void print_statistics() const {
        spdlog::info("=== EDC Statistics ===");
        spdlog::info("Total Predictions: {}, Accuracy: {:.2f}%",
                    total_predictions_, get_prediction_accuracy() * 100.0);
        spdlog::info("Total Drafts: {}, Suppressed: {} ({:.2f}%)",
                    total_drafts_, suppressed_drafts_, 
                    get_suppression_rate() * 100.0);
        spdlog::info("Current LLR: {}", llr_);
        
        // Print LEHT state
        spdlog::info("LEHT: [{}, {}, {}, {}, {}, {}, {}, {}]",
                    leht_[0], leht_[1], leht_[2], leht_[3],
                    leht_[4], leht_[5], leht_[6], leht_[7]);
    }
    
    // Hardware cost estimation (for paper)
    static constexpr size_t get_area_bits() {
        // LEHT: 8 entries × 3 bits = 24 bits
        // LCEHT: 8 entries × 3 bits = 24 bits
        // LLR: 3 bits
        // PHT: 512 entries × 2 bits = 1024 bits
        // Total: ~1075 bits ≈ 135 bytes
        return 24 + 24 + 3 + 1024;
    }
};

} // namespace AHASD

