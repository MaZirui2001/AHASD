package xiangshan.ahasd

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import xiangshan._

/**
 * AHASD Control Module
 * 
 * This module implements the CPU-side control logic for AHASD
 * (Asynchronous Heterogeneous Architecture for LLM Speculative Decoding).
 * 
 * Key responsibilities:
 * 1. Poll EDC (Entropy-History-Aware Drafting Control) decisions
 * 2. Poll TVC (Time-Aware Pre-Verification Control) decisions  
 * 3. Manage async queues between NPU and PIM
 * 4. Handle interrupts from hardware accelerators
 * 5. Collect statistics for performance monitoring
 * 
 * Paper reference: "AHASD: Asynchronous Heterogeneous Architecture for 
 * LLM Speculative Decoding on Mobile Devices"
 */

class AHASDControlIO(implicit p: Parameters) extends XSBundle {
  // Memory-mapped register interface
  val mmio = Flipped(new MemoryMappedIO)
  
  // EDC interface
  val edc_should_continue = Output(Bool())
  val edc_entropy_value = Input(UInt(32.W))
  val edc_llr_count = Input(UInt(3.W))
  val edc_update = Output(Bool())
  
  // TVC interface  
  val tvc_should_preverify = Output(Bool())
  val tvc_preverify_length = Output(UInt(8.W))
  val tvc_npu_cycles = Input(UInt(64.W))
  val tvc_pim_cycles = Input(UInt(64.W))
  
  // Async queue status
  val draft_queue_count = Input(UInt(8.W))
  val feedback_queue_count = Input(UInt(8.W))
  val preverify_queue_count = Input(UInt(8.W))
  
  // Interrupt signals
  val queue_overflow_irq = Output(Bool())
  val edc_decision_irq = Output(Bool())
  val tvc_decision_irq = Output(Bool())
}

class MemoryMappedIO extends Bundle {
  val addr = Input(UInt(32.W))
  val data_write = Input(UInt(64.W))
  val data_read = Output(UInt(64.W))
  val write_enable = Input(Bool())
  val read_enable = Input(Bool())
  val ready = Output(Bool())
}

class AHASDControl(implicit p: Parameters) extends XSModule {
  val io = IO(new AHASDControlIO)
  
  // ========== Configuration Registers ==========
  // These match the configuration file: ahasd_control_config.txt
  
  val cfg_enable_ahasd = RegInit(true.B)
  val cfg_edc_poll_interval = RegInit(100.U(32.W))  // CPU cycles
  val cfg_tvc_measure_interval = RegInit(50.U(32.W))
  val cfg_queue_poll_interval = RegInit(20.U(32.W))
  val cfg_overflow_threshold = RegInit(56.U(8.W))
  val cfg_max_entries_per_poll = RegInit(4.U(8.W))
  
  // ========== State Registers ==========
  
  // Polling timers
  val edc_poll_timer = RegInit(0.U(32.W))
  val tvc_measure_timer = RegInit(0.U(32.W))
  val queue_poll_timer = RegInit(0.U(32.W))
  
  // EDC state tracking
  val edc_last_decision = RegInit(true.B)
  val edc_decision_count = RegInit(0.U(64.W))
  val edc_suppress_count = RegInit(0.U(64.W))
  
  // TVC state tracking
  val tvc_last_preverify_len = RegInit(0.U(8.W))
  val tvc_preverify_count = RegInit(0.U(64.W))
  val tvc_prevented_idles = RegInit(0.U(64.W))
  
  // Queue statistics
  val draft_queue_max = RegInit(0.U(8.W))
  val feedback_queue_max = RegInit(0.U(8.W))
  val preverify_queue_max = RegInit(0.U(8.W))
  val total_queue_overflows = RegInit(0.U(32.W))
  
  // Interrupt flags
  val queue_overflow_flag = RegInit(false.B)
  val edc_decision_flag = RegInit(false.B)
  val tvc_decision_flag = RegInit(false.B)
  
  // ========== EDC Polling Logic ==========
  
  // Increment EDC poll timer
  when(cfg_enable_ahasd) {
    when(edc_poll_timer >= cfg_edc_poll_interval) {
      edc_poll_timer := 0.U
      
      // Trigger EDC decision query
      io.edc_update := true.B
      edc_decision_count := edc_decision_count + 1.U
      
      // Track suppression rate
      when(!io.edc_should_continue) {
        edc_suppress_count := edc_suppress_count + 1.U
      }
      
      // Update decision flag for interrupt
      when(edc_last_decision =/= io.edc_should_continue) {
        edc_decision_flag := true.B
      }
      edc_last_decision := io.edc_should_continue
      
    }.otherwise {
      edc_poll_timer := edc_poll_timer + 1.U
      io.edc_update := false.B
    }
  }.otherwise {
    io.edc_update := false.B
  }
  
  // ========== TVC Polling Logic ==========
  
  // Increment TVC measurement timer
  when(cfg_enable_ahasd) {
    when(tvc_measure_timer >= cfg_tvc_measure_interval) {
      tvc_measure_timer := 0.U
      
      // Calculate if pre-verification should be inserted
      // This is a simplified version - actual logic is in hardware TVC module
      val npu_ahead = io.tvc_npu_cycles > io.tvc_pim_cycles
      val has_pending_drafts = io.draft_queue_count > 0.U
      val should_preverify = npu_ahead && has_pending_drafts && 
                            (io.draft_queue_count >= 2.U)
      
      io.tvc_should_preverify := should_preverify
      
      when(should_preverify) {
        // Conservative pre-verification length (2-4 tokens)
        io.tvc_preverify_length := Mux(io.draft_queue_count >= 4.U, 4.U, 2.U)
        tvc_last_preverify_len := io.tvc_preverify_length
        tvc_preverify_count := tvc_preverify_count + 1.U
        tvc_decision_flag := true.B
      }.otherwise {
        io.tvc_preverify_length := 0.U
      }
      
    }.otherwise {
      tvc_measure_timer := tvc_measure_timer + 1.U
      io.tvc_should_preverify := false.B
      io.tvc_preverify_length := 0.U
    }
  }.otherwise {
    io.tvc_should_preverify := false.B
    io.tvc_preverify_length := 0.U
  }
  
  // ========== Queue Management ==========
  
  // Poll async queues for overflow detection
  when(cfg_enable_ahasd) {
    when(queue_poll_timer >= cfg_queue_poll_interval) {
      queue_poll_timer := 0.U
      
      // Update max queue depths
      when(io.draft_queue_count > draft_queue_max) {
        draft_queue_max := io.draft_queue_count
      }
      when(io.feedback_queue_count > feedback_queue_max) {
        feedback_queue_max := io.feedback_queue_count
      }
      when(io.preverify_queue_count > preverify_queue_max) {
        preverify_queue_max := io.preverify_queue_count
      }
      
      // Check for overflow
      val any_overflow = (io.draft_queue_count >= cfg_overflow_threshold) ||
                        (io.feedback_queue_count >= cfg_overflow_threshold) ||
                        (io.preverify_queue_count >= cfg_overflow_threshold)
      
      when(any_overflow) {
        queue_overflow_flag := true.B
        total_queue_overflows := total_queue_overflows + 1.U
      }
      
    }.otherwise {
      queue_poll_timer := queue_poll_timer + 1.U
    }
  }
  
  // ========== Interrupt Output ==========
  
  io.queue_overflow_irq := queue_overflow_flag
  io.edc_decision_irq := edc_decision_flag
  io.tvc_decision_irq := tvc_decision_flag
  
  // ========== Memory-Mapped Register Interface ==========
  
  // Register address map (byte addresses)
  val REG_CONTROL         = 0x00.U  // Control register (enable/disable)
  val REG_EDC_INTERVAL    = 0x08.U  // EDC poll interval
  val REG_TVC_INTERVAL    = 0x10.U  // TVC measure interval
  val REG_QUEUE_INTERVAL  = 0x18.U  // Queue poll interval
  val REG_EDC_DECISION    = 0x20.U  // Last EDC decision
  val REG_EDC_SUPPRESS    = 0x28.U  // EDC suppression count
  val REG_TVC_PREVERIFY   = 0x30.U  // TVC pre-verification count
  val REG_TVC_PREVENTED   = 0x38.U  // TVC prevented idles
  val REG_DRAFT_QUEUE     = 0x40.U  // Draft queue depth
  val REG_FEEDBACK_QUEUE  = 0x48.U  // Feedback queue depth
  val REG_PREVERIFY_QUEUE = 0x50.U  // Pre-verify queue depth
  val REG_QUEUE_OVERFLOWS = 0x58.U  // Total queue overflows
  val REG_INTERRUPT_FLAGS = 0x60.U  // Interrupt flags
  val REG_INTERRUPT_CLEAR = 0x68.U  // Interrupt clear (write-only)
  
  // Default MMIO output
  io.mmio.data_read := 0.U
  io.mmio.ready := true.B
  
  // Handle MMIO reads
  when(io.mmio.read_enable) {
    switch(io.mmio.addr) {
      is(REG_CONTROL) {
        io.mmio.data_read := cfg_enable_ahasd.asUInt
      }
      is(REG_EDC_INTERVAL) {
        io.mmio.data_read := cfg_edc_poll_interval
      }
      is(REG_TVC_INTERVAL) {
        io.mmio.data_read := cfg_tvc_measure_interval
      }
      is(REG_QUEUE_INTERVAL) {
        io.mmio.data_read := cfg_queue_poll_interval
      }
      is(REG_EDC_DECISION) {
        io.mmio.data_read := edc_last_decision.asUInt
      }
      is(REG_EDC_SUPPRESS) {
        io.mmio.data_read := edc_suppress_count
      }
      is(REG_TVC_PREVERIFY) {
        io.mmio.data_read := tvc_preverify_count
      }
      is(REG_TVC_PREVENTED) {
        io.mmio.data_read := tvc_prevented_idles
      }
      is(REG_DRAFT_QUEUE) {
        io.mmio.data_read := io.draft_queue_count
      }
      is(REG_FEEDBACK_QUEUE) {
        io.mmio.data_read := io.feedback_queue_count
      }
      is(REG_PREVERIFY_QUEUE) {
        io.mmio.data_read := io.preverify_queue_count
      }
      is(REG_QUEUE_OVERFLOWS) {
        io.mmio.data_read := total_queue_overflows
      }
      is(REG_INTERRUPT_FLAGS) {
        val flags = Cat(tvc_decision_flag, edc_decision_flag, queue_overflow_flag)
        io.mmio.data_read := flags
      }
    }
  }
  
  // Handle MMIO writes
  when(io.mmio.write_enable) {
    switch(io.mmio.addr) {
      is(REG_CONTROL) {
        cfg_enable_ahasd := io.mmio.data_write(0)
      }
      is(REG_EDC_INTERVAL) {
        cfg_edc_poll_interval := io.mmio.data_write(31, 0)
      }
      is(REG_TVC_INTERVAL) {
        cfg_tvc_measure_interval := io.mmio.data_write(31, 0)
      }
      is(REG_QUEUE_INTERVAL) {
        cfg_queue_poll_interval := io.mmio.data_write(31, 0)
      }
      is(REG_INTERRUPT_CLEAR) {
        // Clear specified interrupt flags
        when(io.mmio.data_write(0)) {
          queue_overflow_flag := false.B
        }
        when(io.mmio.data_write(1)) {
          edc_decision_flag := false.B
        }
        when(io.mmio.data_write(2)) {
          tvc_decision_flag := false.B
        }
      }
    }
  }
}

/**
 * Companion object for AHASDControl
 * Provides utility functions and constants
 */
object AHASDControl {
  // Register address constants for software access
  val REG_MAP = Map(
    "CONTROL" -> 0x00,
    "EDC_INTERVAL" -> 0x08,
    "TVC_INTERVAL" -> 0x10,
    "QUEUE_INTERVAL" -> 0x18,
    "EDC_DECISION" -> 0x20,
    "EDC_SUPPRESS" -> 0x28,
    "TVC_PREVERIFY" -> 0x30,
    "TVC_PREVENTED" -> 0x38,
    "DRAFT_QUEUE" -> 0x40,
    "FEEDBACK_QUEUE" -> 0x48,
    "PREVERIFY_QUEUE" -> 0x50,
    "QUEUE_OVERFLOWS" -> 0x58,
    "INTERRUPT_FLAGS" -> 0x60,
    "INTERRUPT_CLEAR" -> 0x68
  )
  
  // Default configuration values
  val DEFAULT_EDC_POLL_INTERVAL = 100
  val DEFAULT_TVC_MEASURE_INTERVAL = 50
  val DEFAULT_QUEUE_POLL_INTERVAL = 20
  val DEFAULT_OVERFLOW_THRESHOLD = 56
  
  def apply()(implicit p: Parameters): AHASDControl = new AHASDControl
}
