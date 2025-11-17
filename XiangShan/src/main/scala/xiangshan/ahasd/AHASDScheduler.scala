package xiangshan.ahasd

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import xiangshan._

/**
 * AHASD Task Scheduler Module
 * 
 * This module implements the CPU-side task scheduling logic for AHASD.
 * It coordinates task dispatching between NPU (for TLM verification) 
 * and PIM (for DLM drafting and pre-verification).
 * 
 * Key responsibilities:
 * 1. Dispatch drafting tasks to PIM based on EDC decisions
 * 2. Dispatch verification tasks to NPU when drafts are ready
 * 3. Insert pre-verification tasks on PIM based on TVC decisions
 * 4. Handle task completion and feedback
 * 5. Maintain task-level asynchronous execution
 * 
 * Paper reference: Section 4.1 "AHASD Task-Level Asynchronous 
 * Heterogeneous Execution Framework"
 */

class AHASDSchedulerIO(implicit p: Parameters) extends XSBundle {
  // Control interface
  val enable = Input(Bool())
  
  // EDC interface
  val edc_should_continue = Input(Bool())
  val edc_current_entropy = Input(UInt(32.W))
  
  // TVC interface
  val tvc_should_preverify = Input(Bool())
  val tvc_preverify_length = Input(UInt(8.W))
  
  // NPU task interface
  val npu_task_valid = Output(Bool())
  val npu_task_ready = Input(Bool())
  val npu_task_type = Output(TaskType())
  val npu_task_length = Output(UInt(16.W))
  val npu_task_id = Output(UInt(32.W))
  
  val npu_complete_valid = Input(Bool())
  val npu_complete_id = Input(UInt(32.W))
  val npu_complete_cycles = Input(UInt(64.W))
  val npu_complete_accepted = Input(UInt(16.W))
  
  // PIM task interface
  val pim_task_valid = Output(Bool())
  val pim_task_ready = Input(Bool())
  val pim_task_type = Output(TaskType())
  val pim_task_length = Output(UInt(16.W))
  val pim_task_id = Output(UInt(32.W))
  
  val pim_complete_valid = Input(Bool())
  val pim_complete_id = Input(UInt(32.W))
  val pim_complete_cycles = Input(UInt(64.W))
  val pim_complete_entropy = Input(UInt(32.W))
  
  // Async queue interfaces
  val draft_queue_push = Output(Bool())
  val draft_queue_full = Input(Bool())
  val draft_queue_count = Input(UInt(8.W))
  
  val feedback_queue_pop = Output(Bool())
  val feedback_queue_empty = Input(Bool())
  
  // Statistics
  val stat_total_drafts = Output(UInt(64.W))
  val stat_total_verifications = Output(UInt(64.W))
  val stat_total_preverifications = Output(UInt(64.W))
  val stat_suppressed_drafts = Output(UInt(64.W))
}

class TaskType extends Bundle {
  val is_draft = Bool()
  val is_verify = Bool()
  val is_preverify = Bool()
}

object TaskType {
  def draft = {
    val t = Wire(new TaskType)
    t.is_draft := true.B
    t.is_verify := false.B
    t.is_preverify := false.B
    t
  }
  
  def verify = {
    val t = Wire(new TaskType)
    t.is_draft := false.B
    t.is_verify := true.B
    t.is_preverify := false.B
    t
  }
  
  def preverify = {
    val t = Wire(new TaskType)
    t.is_draft := false.B
    t.is_verify := false.B
    t.is_preverify := true.B
    t
  }
}

class AHASDScheduler(implicit p: Parameters) extends XSModule {
  val io = IO(new AHASDSchedulerIO)
  
  // ========== Task State Machine ==========
  
  val sIdle :: sDrafting :: sVerifying :: sPreVerifying :: sWaitFeedback :: Nil = Enum(5)
  val pim_state = RegInit(sIdle)
  val npu_state = RegInit(sIdle)
  
  // ========== Task ID Generators ==========
  
  val next_draft_id = RegInit(0.U(32.W))
  val next_verify_id = RegInit(0.U(32.W))
  val next_preverify_id = RegInit(0.U(32.W))
  
  // ========== Configuration ==========
  
  val cfg_max_draft_length = RegInit(16.U(16.W))
  val cfg_min_preverify_length = RegInit(2.U(16.W))
  val cfg_max_pending_drafts = RegInit(8.U(8.W))
  
  // ========== Statistics Counters ==========
  
  val total_drafts = RegInit(0.U(64.W))
  val total_verifications = RegInit(0.U(64.W))
  val total_preverifications = RegInit(0.U(64.W))
  val suppressed_drafts = RegInit(0.U(64.W))
  val npu_idle_cycles = RegInit(0.U(64.W))
  val pim_idle_cycles = RegInit(0.U(64.W))
  
  // Output statistics
  io.stat_total_drafts := total_drafts
  io.stat_total_verifications := total_verifications
  io.stat_total_preverifications := total_preverifications
  io.stat_suppressed_drafts := suppressed_drafts
  
  // ========== Current Task Tracking ==========
  
  val current_draft_id = RegInit(0.U(32.W))
  val current_draft_length = RegInit(0.U(16.W))
  val current_draft_entropy = RegInit(0.U(32.W))
  
  val current_verify_id = RegInit(0.U(32.W))
  val current_verify_length = RegInit(0.U(16.W))
  
  val current_preverify_id = RegInit(0.U(32.W))
  val current_preverify_length = RegInit(0.U(16.W))
  
  // ========== PIM-side Task Scheduling ==========
  
  // Default outputs
  io.pim_task_valid := false.B
  io.pim_task_type := TaskType.draft
  io.pim_task_length := 0.U
  io.pim_task_id := 0.U
  io.draft_queue_push := false.B
  
  switch(pim_state) {
    is(sIdle) {
      when(io.enable) {
        // Check if should start drafting
        val can_draft = !io.draft_queue_full && 
                       (io.draft_queue_count < cfg_max_pending_drafts)
        
        when(can_draft && io.edc_should_continue) {
          // Start new draft task
          pim_state := sDrafting
          current_draft_id := next_draft_id
          next_draft_id := next_draft_id + 1.U
          
          // Dispatch draft task to PIM
          io.pim_task_valid := true.B
          io.pim_task_type := TaskType.draft
          io.pim_task_length := cfg_max_draft_length
          io.pim_task_id := current_draft_id
          
          when(io.pim_task_ready) {
            total_drafts := total_drafts + 1.U
          }
          
        }.elsewhen(!io.edc_should_continue) {
          // EDC suppressed drafting
          suppressed_drafts := suppressed_drafts + 1.U
          
          // Check if TVC suggests pre-verification
          when(io.tvc_should_preverify && io.draft_queue_count > 0.U) {
            pim_state := sPreVerifying
            current_preverify_id := next_preverify_id
            next_preverify_id := next_preverify_id + 1.U
            current_preverify_length := io.tvc_preverify_length
            
            // Dispatch pre-verify task to PIM
            io.pim_task_valid := true.B
            io.pim_task_type := TaskType.preverify
            io.pim_task_length := io.tvc_preverify_length
            io.pim_task_id := current_preverify_id
            
            when(io.pim_task_ready) {
              total_preverifications := total_preverifications + 1.U
            }
          }.otherwise {
            // PIM remains idle
            pim_idle_cycles := pim_idle_cycles + 1.U
          }
        }.otherwise {
          pim_idle_cycles := pim_idle_cycles + 1.U
        }
      }
    }
    
    is(sDrafting) {
      // Wait for draft completion
      when(io.pim_complete_valid && io.pim_complete_id === current_draft_id) {
        // Draft completed, push to queue
        io.draft_queue_push := true.B
        current_draft_entropy := io.pim_complete_entropy
        
        // Return to idle to check for next task
        pim_state := sIdle
      }
    }
    
    is(sPreVerifying) {
      // Wait for pre-verification completion
      when(io.pim_complete_valid && io.pim_complete_id === current_preverify_id) {
        // Pre-verification completed
        // Results are automatically fed back to queue management
        
        // Return to idle
        pim_state := sIdle
      }
    }
  }
  
  // ========== NPU-side Task Scheduling ==========
  
  // Default outputs
  io.npu_task_valid := false.B
  io.npu_task_type := TaskType.verify
  io.npu_task_length := 0.U
  io.npu_task_id := 0.U
  io.feedback_queue_pop := false.B
  
  switch(npu_state) {
    is(sIdle) {
      when(io.enable) {
        // Check if drafts are available for verification
        when(io.draft_queue_count > 0.U) {
          // Start verification task
          npu_state := sVerifying
          current_verify_id := next_verify_id
          next_verify_id := next_verify_id + 1.U
          current_verify_length := io.draft_queue_count  // Verify all pending
          
          // Dispatch verify task to NPU
          io.npu_task_valid := true.B
          io.npu_task_type := TaskType.verify
          io.npu_task_length := current_verify_length
          io.npu_task_id := current_verify_id
          
          when(io.npu_task_ready) {
            total_verifications := total_verifications + 1.U
          }
          
        }.otherwise {
          // NPU idle - waiting for drafts
          npu_idle_cycles := npu_idle_cycles + 1.U
        }
      }
    }
    
    is(sVerifying) {
      // Wait for verification completion
      when(io.npu_complete_valid && io.npu_complete_id === current_verify_id) {
        // Verification completed
        // Feedback is automatically sent through feedback queue
        
        npu_state := sWaitFeedback
      }
    }
    
    is(sWaitFeedback) {
      // Wait for feedback to be consumed by PIM
      when(!io.feedback_queue_empty) {
        io.feedback_queue_pop := true.B
        npu_state := sIdle
      }.otherwise {
        // Still waiting
        npu_state := sIdle  // Can proceed to next verification
      }
    }
  }
  
  // ========== Task-Level Asynchronous Execution ==========
  
  // Key insight: PIM and NPU state machines run independently
  // They communicate only through async queues (draft_queue, feedback_queue)
  // This implements the task-level asynchronous parallelism described in the paper
  
  // PIM progresses at its own pace: sIdle -> sDrafting -> sIdle (or sPreVerifying)
  // NPU progresses at its own pace: sIdle -> sVerifying -> sWaitFeedback -> sIdle
  
  // No explicit synchronization barriers between PIM and NPU
  // This addresses Challenge 1 from the paper (operator-level sync overhead)
  
  // ========== Debug and Monitoring ==========
  
  when(io.enable) {
    // Print debug information periodically (every 10000 cycles)
    val debug_counter = RegInit(0.U(32.W))
    debug_counter := debug_counter + 1.U
    
    when(debug_counter === 10000.U) {
      debug_counter := 0.U
      
      printf(p"[AHASD Scheduler] Drafts: ${total_drafts}, " +
             p"Verifications: ${total_verifications}, " +
             p"PreVerify: ${total_preverifications}, " +
             p"Suppressed: ${suppressed_drafts}\n")
      
      printf(p"[AHASD Scheduler] PIM State: ${pim_state}, " +
             p"NPU State: ${npu_state}, " +
             p"Queue: ${io.draft_queue_count}\n")
    }
  }
}

/**
 * Companion object for AHASDScheduler
 */
object AHASDScheduler {
  def apply()(implicit p: Parameters): AHASDScheduler = new AHASDScheduler
}
