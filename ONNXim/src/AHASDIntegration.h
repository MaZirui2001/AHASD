#pragma once

#include "Common.h"
#include "async_queue/AsyncQueue.h"
#include "async_queue/EDC.h"
#include "async_queue/TVC.h"
#include <memory>
#include <fstream>

// AHASD Integration Layer
// Coordinates NPU-side and PIM-side operations for speculative decoding

namespace AHASD {

struct AHASDConfig {
    bool enable_edc;  // Enable Entropy-History-Aware Drafting Control
    bool enable_tvc;  // Enable Time-Aware Pre-Verification Control
    bool enable_aau;  // Enable Attention Algorithm Unit
    float pim_freq_mhz;
    float npu_freq_mhz;
    uint32_t max_draft_length;
    uint32_t min_preverify_length;
    
    AHASDConfig() 
        : enable_edc(true), enable_tvc(true), enable_aau(true),
          pim_freq_mhz(800.0f), npu_freq_mhz(1000.0f),
          max_draft_length(16), min_preverify_length(2) {}
};

class AHASDIntegration {
private:
    AHASDConfig config_;
    
    // Core components
    std::unique_ptr<AsyncQueueManager> queue_manager_;
    std::unique_ptr<EDC> edc_;
    std::unique_ptr<TVC> tvc_;
    
    // State tracking
    uint32_t current_kv_length_;
    uint32_t current_batch_id_;
    bool npu_busy_;
    bool pim_busy_;
    
    // Performance statistics
    uint64_t total_drafts_generated_;
    uint64_t total_drafts_accepted_;
    uint64_t total_preverifications_;
    uint64_t total_npu_idle_cycles_;
    uint64_t total_pim_idle_cycles_;
    double total_draft_entropy_;
    
    // Timing
    uint64_t last_verification_start_;
    uint64_t last_drafting_start_;
    
    // Logging
    std::ofstream trace_file_;
    bool enable_tracing_;

public:
    AHASDIntegration(const AHASDConfig& config = AHASDConfig())
        : config_(config), current_kv_length_(0), current_batch_id_(0),
          npu_busy_(false), pim_busy_(false),
          total_drafts_generated_(0), total_drafts_accepted_(0),
          total_preverifications_(0), total_npu_idle_cycles_(0),
          total_pim_idle_cycles_(0), total_draft_entropy_(0.0),
          last_verification_start_(0), last_drafting_start_(0),
          enable_tracing_(false) {
        
        queue_manager_ = std::make_unique<AsyncQueueManager>();
        
        if (config_.enable_edc) {
            edc_ = std::make_unique<EDC>();
        }
        
        if (config_.enable_tvc) {
            tvc_ = std::make_unique<TVC>(config_.pim_freq_mhz, config_.npu_freq_mhz);
        }
    }
    
    ~AHASDIntegration() {
        if (trace_file_.is_open()) {
            trace_file_.close();
        }
    }
    
    // Enable trace logging
    void enable_trace_logging(const std::string& filename) {
        trace_file_.open(filename);
        if (trace_file_.is_open()) {
            enable_tracing_ = true;
            trace_file_ << "cycle,event,batch_id,length,entropy,decision\n";
        }
    }
    
    // PIM-side: Generate draft
    bool submit_draft_batch(const std::vector<int32_t>& tokens,
                           const std::vector<float>& entropies,
                           uint64_t cycle) {
        DraftBatch batch;
        batch.batch_id = current_batch_id_++;
        batch.draft_length = tokens.size();
        batch.token_ids = tokens;
        batch.entropy_values = entropies;
        batch.timestamp = cycle;
        batch.verified = false;
        batch.accepted = false;
        
        // Calculate average entropy
        float avg_entropy = 0.0f;
        for (float e : entropies) {
            avg_entropy += e;
        }
        if (!entropies.empty()) {
            avg_entropy /= entropies.size();
        }
        total_draft_entropy_ += avg_entropy;
        
        bool success = queue_manager_->push_draft(batch);
        if (success) {
            total_drafts_generated_++;
            
            if (enable_tracing_) {
                trace_file_ << cycle << ",draft_generated," << batch.batch_id 
                           << "," << batch.draft_length << "," << avg_entropy 
                           << ",NA\n";
            }
        }
        
        return success;
    }
    
    // PIM-side: Check if should continue drafting
    bool should_continue_drafting(float avg_entropy) {
        if (!config_.enable_edc || edc_ == nullptr) {
            // Without EDC, always continue up to max length
            return queue_manager_->get_unverified_count() < config_.max_draft_length;
        }
        
        bool edc_decision = edc_->should_continue_drafting(avg_entropy);
        
        // Check TVC for pre-verification opportunity
        if (!edc_decision && config_.enable_tvc && tvc_ != nullptr) {
            uint32_t pending = queue_manager_->get_unverified_count();
            if (pending >= config_.min_preverify_length) {
                auto [should_preverify, length] = tvc_->should_insert_preverification(
                    current_kv_length_, pending);
                
                if (should_preverify) {
                    // Submit pre-verification request
                    PreVerifyRequest req;
                    req.verify_length = length;
                    req.timestamp = queue_manager_->get_pim_cycles();
                    req.urgent = false;
                    queue_manager_->push_preverify_request(req);
                    total_preverifications_++;
                    
                    if (enable_tracing_) {
                        trace_file_ << queue_manager_->get_pim_cycles() 
                                   << ",preverify_inserted,0," << length 
                                   << ",0.0,tvc\n";
                    }
                }
            }
        }
        
        return edc_decision;
    }
    
    // NPU-side: Pop draft for verification
    bool get_next_draft(DraftBatch& batch) {
        return queue_manager_->pop_draft(batch);
    }
    
    // NPU-side: Submit verification feedback
    void submit_verification_result(uint32_t batch_id, uint32_t accepted_length,
                                    bool fully_accepted, uint64_t verification_cycles,
                                    uint32_t kv_length) {
        FeedbackData feedback;
        feedback.batch_id = batch_id;
        feedback.accepted_length = accepted_length;
        feedback.fully_accepted = fully_accepted;
        feedback.verification_cycles = verification_cycles;
        feedback.kv_cache_length = kv_length;
        
        queue_manager_->push_feedback(feedback);
        
        if (fully_accepted || accepted_length > 0) {
            total_drafts_accepted_ += accepted_length;
        }
        
        current_kv_length_ = kv_length;
        
        // Update EDC
        if (config_.enable_edc && edc_ != nullptr) {
            edc_->update_on_verification(fully_accepted, accepted_length);
        }
        
        // Update TVC
        if (config_.enable_tvc && tvc_ != nullptr) {
            tvc_->record_npu_verification(verification_cycles, kv_length);
        }
        
        if (enable_tracing_) {
            trace_file_ << queue_manager_->get_npu_cycles() 
                       << ",verification_result," << batch_id << "," 
                       << accepted_length << ",0.0," 
                       << (fully_accepted ? "full" : "partial") << "\n";
        }
    }
    
    // PIM-side: Check for feedback
    bool get_feedback(FeedbackData& feedback) {
        return queue_manager_->pop_feedback(feedback);
    }
    
    // PIM-side: Check for pre-verification request
    bool get_preverify_request(PreVerifyRequest& request) {
        return queue_manager_->pop_preverify_request(request);
    }
    
    // Record PIM drafting time
    void record_pim_drafting(uint64_t cycles, uint32_t draft_length) {
        if (config_.enable_tvc && tvc_ != nullptr) {
            tvc_->record_pim_drafting(cycles, draft_length);
        }
    }
    
    // Record PIM pre-verification time
    void record_pim_preverification(uint64_t cycles, uint32_t draft_length) {
        if (config_.enable_tvc && tvc_ != nullptr) {
            tvc_->record_pim_preverification(cycles, draft_length);
        }
    }
    
    // Start NPU verification task
    void start_npu_verification(uint64_t current_cycle) {
        last_verification_start_ = current_cycle;
        npu_busy_ = true;
        
        if (config_.enable_tvc && tvc_ != nullptr) {
            tvc_->start_npu_task(current_cycle);
        }
    }
    
    // Update NPU progress
    void update_npu_progress(uint64_t current_cycle) {
        if (config_.enable_tvc && tvc_ != nullptr && npu_busy_) {
            tvc_->update_npu_progress(current_cycle);
        }
    }
    
    // Finish NPU verification
    void finish_npu_verification() {
        npu_busy_ = false;
    }
    
    // Cycle updates
    void cycle_npu() {
        queue_manager_->increment_npu_cycle();
        if (!npu_busy_ && queue_manager_->has_pending_drafts()) {
            total_npu_idle_cycles_++;
        }
    }
    
    void cycle_pim() {
        queue_manager_->increment_pim_cycle();
        if (!pim_busy_) {
            total_pim_idle_cycles_++;
        }
    }
    
    // Status queries
    bool has_pending_drafts() const {
        return queue_manager_->has_pending_drafts();
    }
    
    size_t get_pending_draft_count() const {
        return queue_manager_->get_unverified_count();
    }
    
    bool is_npu_busy() const { return npu_busy_; }
    bool is_pim_busy() const { return pim_busy_; }
    
    void set_npu_busy(bool busy) { npu_busy_ = busy; }
    void set_pim_busy(bool busy) { pim_busy_ = busy; }
    
    // Statistics
    double get_acceptance_rate() const {
        if (total_drafts_generated_ == 0) return 0.0;
        return static_cast<double>(total_drafts_accepted_) / total_drafts_generated_;
    }
    
    double get_average_entropy() const {
        if (total_drafts_generated_ == 0) return 0.0;
        return total_draft_entropy_ / total_drafts_generated_;
    }
    
    void print_statistics() const {
        spdlog::info("=== AHASD Integration Statistics ===");
        spdlog::info("Total Drafts Generated: {}", total_drafts_generated_);
        spdlog::info("Total Drafts Accepted: {} ({:.2f}%)", 
                    total_drafts_accepted_, get_acceptance_rate() * 100.0);
        spdlog::info("Total Pre-verifications: {}", total_preverifications_);
        spdlog::info("Average Draft Entropy: {:.3f}", get_average_entropy());
        spdlog::info("NPU Idle Cycles: {}", total_npu_idle_cycles_);
        spdlog::info("PIM Idle Cycles: {}", total_pim_idle_cycles_);
        
        queue_manager_->print_statistics();
        
        if (config_.enable_edc && edc_ != nullptr) {
            edc_->print_statistics();
        }
        
        if (config_.enable_tvc && tvc_ != nullptr) {
            tvc_->print_statistics();
        }
    }
    
    // Hardware cost summary for paper
    static void print_hardware_costs() {
        spdlog::info("=== AHASD Hardware Overhead ===");
        
        size_t edc_bits = EDC::get_area_bits();
        size_t tvc_bits = TVC::get_area_bits();
        size_t async_queue_bits = 3 * 1024;  // 3 queues, ~1KB each
        
        double edc_mm2 = edc_bits / (8.0 * 1024 * 1024) * 100;  // Rough estimate
        double tvc_mm2 = tvc_bits / (8.0 * 1024 * 1024) * 100;
        double queue_mm2 = 0.001;  // Minimal SRAM
        double aau_mm2 = 0.45;  // From AAU spec
        
        double total_mm2 = edc_mm2 + tvc_mm2 + queue_mm2 + aau_mm2;
        double lpddr5_die_mm2 = 18.0;  // Typical LPDDR5 die size
        
        spdlog::info("EDC: {:.4f} mm² ({} bits)", edc_mm2, edc_bits);
        spdlog::info("TVC: {:.4f} mm² ({} bits)", tvc_mm2, tvc_bits);
        spdlog::info("Async Queues: {:.4f} mm²", queue_mm2);
        spdlog::info("AAU: {:.2f} mm²", aau_mm2);
        spdlog::info("Total: {:.3f} mm² ({:.2f}% of LPDDR5 die)",
                    total_mm2, (total_mm2 / lpddr5_die_mm2) * 100.0);
    }
    
    // Reset for new inference sequence
    void reset() {
        current_kv_length_ = 0;
        current_batch_id_ = 0;
        npu_busy_ = false;
        pim_busy_ = false;
        
        if (edc_ != nullptr) {
            edc_->reset();
        }
        
        if (tvc_ != nullptr) {
            tvc_->reset();
        }
    }
};

} // namespace AHASD

