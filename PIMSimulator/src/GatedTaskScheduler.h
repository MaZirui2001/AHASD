/*********************************************************************************
 * Gated Task Scheduling Unit for LPDDR5-PIM
 * Enables sub-microsecond task switching between drafting and pre-verification
 * Uses rank-level gating within the same PIM array
 *********************************************************************************/

#ifndef GATED_TASK_SCHEDULER_H
#define GATED_TASK_SCHEDULER_H

#include <cstdint>
#include <vector>
#include <string>

namespace DRAMSim {

enum class PIMTaskType {
    IDLE,
    DRAFTING,           // DLM draft generation
    PRE_VERIFICATION    // Small-batch TLM pre-verification
};

struct TaskDescriptor {
    PIMTaskType type;
    uint32_t rank_id;
    uint32_t batch_size;
    uint64_t start_cycle;
    uint64_t estimated_cycles;
    bool completed;
    
    TaskDescriptor() : type(PIMTaskType::IDLE), rank_id(0), batch_size(0),
                      start_cycle(0), estimated_cycles(0), completed(false) {}
};

struct RankState {
    bool enabled;
    bool busy;
    PIMTaskType current_task;
    uint32_t remaining_cycles;
    
    RankState() : enabled(false), busy(false), 
                 current_task(PIMTaskType::IDLE), remaining_cycles(0) {}
};

class GatedTaskScheduler {
private:
    uint32_t num_ranks_;
    std::vector<RankState> rank_states_;
    
    // Task queue
    std::vector<TaskDescriptor> pending_tasks_;
    TaskDescriptor* active_task_;
    
    // Gating control
    uint32_t drafting_rank_;       // Rank(s) for DLM drafting
    uint32_t verification_rank_;   // Rank for TLM pre-verification
    
    // Task switching overhead
    uint32_t switch_latency_cycles_;  // Sub-microsecond switching
    uint32_t current_switch_delay_;
    bool switching_;
    
    // Statistics
    uint64_t total_switches_;
    uint64_t drafting_cycles_;
    uint64_t verification_cycles_;
    uint64_t idle_cycles_;
    uint64_t switch_overhead_cycles_;
    uint64_t total_cycles_;
    
    // Energy tracking
    double total_switch_energy_nj_;
    
public:
    GatedTaskScheduler(uint32_t num_ranks = 16) 
        : num_ranks_(num_ranks), active_task_(nullptr),
          drafting_rank_(0), verification_rank_(num_ranks - 1),
          switch_latency_cycles_(1),  // Sub-microsecond at 800MHz = ~1 cycle
          current_switch_delay_(0), switching_(false),
          total_switches_(0), drafting_cycles_(0), verification_cycles_(0),
          idle_cycles_(0), switch_overhead_cycles_(0), total_cycles_(0),
          total_switch_energy_nj_(0.0) {
        
        rank_states_.resize(num_ranks_);
        
        // Initially enable drafting ranks (most ranks for DLM parameters)
        for (uint32_t i = 0; i < num_ranks_ - 1; i++) {
            rank_states_[i].enabled = true;
        }
        // Reserve last rank for verification parameters
        rank_states_[verification_rank_].enabled = false;
    }
    
    // Submit a task
    bool submit_task(PIMTaskType type, uint32_t batch_size, 
                     uint64_t estimated_cycles) {
        TaskDescriptor task;
        task.type = type;
        task.batch_size = batch_size;
        task.estimated_cycles = estimated_cycles;
        task.completed = false;
        
        pending_tasks_.push_back(task);
        return true;
    }
    
    // Try to schedule next task
    bool schedule_next_task(uint64_t current_cycle) {
        if (active_task_ != nullptr || switching_) {
            return false;  // Already busy
        }
        
        if (pending_tasks_.empty()) {
            return false;  // No pending tasks
        }
        
        TaskDescriptor& next_task = pending_tasks_.front();
        
        // Check if we need to switch rank configuration
        bool need_switch = false;
        if (next_task.type == PIMTaskType::PRE_VERIFICATION) {
            // Need to enable verification rank, disable drafting ranks
            if (rank_states_[verification_rank_].enabled == false) {
                need_switch = true;
            }
        } else if (next_task.type == PIMTaskType::DRAFTING) {
            // Need to enable drafting ranks, disable verification rank
            if (rank_states_[drafting_rank_].enabled == false) {
                need_switch = true;
            }
        }
        
        if (need_switch) {
            initiate_task_switch(next_task.type);
            return true;
        }
        
        // Start task immediately
        start_task(next_task, current_cycle);
        return true;
    }
    
    // Initiate task switching
    void initiate_task_switch(PIMTaskType target_type) {
        switching_ = true;
        current_switch_delay_ = switch_latency_cycles_;
        total_switches_++;
        
        // Gating energy: ~50 pJ per switch
        total_switch_energy_nj_ += 0.05;
        
        // Configure rank enables based on target task
        if (target_type == PIMTaskType::PRE_VERIFICATION) {
            // Disable drafting ranks, enable verification rank
            for (uint32_t i = 0; i < num_ranks_ - 1; i++) {
                rank_states_[i].enabled = false;
            }
            rank_states_[verification_rank_].enabled = true;
        } else {
            // Enable drafting ranks, disable verification rank
            for (uint32_t i = 0; i < num_ranks_ - 1; i++) {
                rank_states_[i].enabled = true;
            }
            rank_states_[verification_rank_].enabled = false;
        }
    }
    
    // Start executing a task
    void start_task(TaskDescriptor& task, uint64_t current_cycle) {
        task.start_cycle = current_cycle;
        active_task_ = &task;
        
        // Mark appropriate ranks as busy
        if (task.type == PIMTaskType::DRAFTING) {
            for (uint32_t i = 0; i < num_ranks_ - 1; i++) {
                rank_states_[i].busy = true;
                rank_states_[i].current_task = PIMTaskType::DRAFTING;
                rank_states_[i].remaining_cycles = task.estimated_cycles;
            }
        } else if (task.type == PIMTaskType::PRE_VERIFICATION) {
            rank_states_[verification_rank_].busy = true;
            rank_states_[verification_rank_].current_task = PIMTaskType::PRE_VERIFICATION;
            rank_states_[verification_rank_].remaining_cycles = task.estimated_cycles;
        }
    }
    
    // Update per cycle
    void update() {
        total_cycles_++;
        
        // Handle switching delay
        if (switching_) {
            if (current_switch_delay_ > 0) {
                current_switch_delay_--;
                switch_overhead_cycles_++;
            } else {
                switching_ = false;
                // Switch complete, start pending task
                if (!pending_tasks_.empty()) {
                    start_task(pending_tasks_.front(), total_cycles_);
                }
            }
            return;
        }
        
        // Update active task
        if (active_task_ != nullptr) {
            bool task_done = true;
            
            for (uint32_t i = 0; i < num_ranks_; i++) {
                if (rank_states_[i].busy) {
                    if (rank_states_[i].remaining_cycles > 0) {
                        rank_states_[i].remaining_cycles--;
                        task_done = false;
                        
                        // Count cycles by task type
                        if (rank_states_[i].current_task == PIMTaskType::DRAFTING) {
                            drafting_cycles_++;
                        } else if (rank_states_[i].current_task == PIMTaskType::PRE_VERIFICATION) {
                            verification_cycles_++;
                        }
                    } else {
                        rank_states_[i].busy = false;
                        rank_states_[i].current_task = PIMTaskType::IDLE;
                    }
                }
            }
            
            if (task_done) {
                active_task_->completed = true;
                active_task_ = nullptr;
                
                // Remove completed task
                if (!pending_tasks_.empty()) {
                    pending_tasks_.erase(pending_tasks_.begin());
                }
            }
        } else {
            idle_cycles_++;
        }
    }
    
    bool is_busy() const {
        return active_task_ != nullptr || switching_;
    }
    
    bool can_accept_task() const {
        return pending_tasks_.size() < 8;  // Queue depth limit
    }
    
    PIMTaskType get_current_task_type() const {
        if (active_task_ != nullptr) {
            return active_task_->type;
        }
        return PIMTaskType::IDLE;
    }
    
    // Statistics
    uint64_t get_total_switches() const { return total_switches_; }
    
    double get_utilization() const {
        if (total_cycles_ == 0) return 0.0;
        uint64_t active_cycles = drafting_cycles_ + verification_cycles_;
        return static_cast<double>(active_cycles) / total_cycles_;
    }
    
    double get_switch_overhead_percent() const {
        if (total_cycles_ == 0) return 0.0;
        return static_cast<double>(switch_overhead_cycles_) / total_cycles_ * 100.0;
    }
    
    void print_stats() const {
        std::cout << "=== Gated Task Scheduler Statistics ===" << std::endl;
        std::cout << "Total Switches: " << total_switches_ << std::endl;
        std::cout << "Drafting Cycles: " << drafting_cycles_ << std::endl;
        std::cout << "Pre-verification Cycles: " << verification_cycles_ << std::endl;
        std::cout << "Idle Cycles: " << idle_cycles_ << std::endl;
        std::cout << "Switch Overhead: " << switch_overhead_cycles_ 
                  << " (" << get_switch_overhead_percent() << "%)" << std::endl;
        std::cout << "Utilization: " << (get_utilization() * 100.0) << "%" << std::endl;
        std::cout << "Total Switch Energy: " << total_switch_energy_nj_ << " nJ" << std::endl;
    }
    
    // Hardware cost estimation
    static constexpr double get_area_mm2() {
        // Gating control logic + rank select mux
        // Minimal overhead: ~0.02 mm^2
        return 0.02;
    }
    
    static constexpr double get_power_mw() {
        // Static power for gating logic
        return 0.5;  // mW
    }
    
    // Verify sub-microsecond switching claim
    static constexpr double get_switch_time_ns(double freq_mhz = 800.0) {
        // switch_latency_cycles = 1
        // Time = cycles / freq = 1 / 800MHz = 1.25 ns
        return (1.0 / freq_mhz) * 1000.0;
    }
};

}  // namespace DRAMSim

#endif  // GATED_TASK_SCHEDULER_H

