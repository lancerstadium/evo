//! `evo::ir::cpu` : IR Context
//! 
//!


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::arch::evo::def::EVO_ARCH;
use crate::arch::riscv::def::RISCV32_ARCH;
use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::Arch;
use crate::core::val::Value;
use crate::core::insn::Instruction;
use crate::core::mem::CPUProcess;
use crate::core::itp::Interpreter;
use crate::core::mem::CPUThreadStatus;

use super::insn::RegFile;


// ============================================================================== //
//                              cpu::CPUState
// ============================================================================== //


/// `CPUState`: Context of the `evo-ir` architecture
#[derive(Clone, PartialEq)]
pub struct CPUState {
    pub src_arch: &'static Arch,
    /// IR Architecture
    pub ir_arch: &'static Arch,
    /// `proc`: Process Handle
    pub proc: Rc<RefCell<CPUProcess>>,
    /// `itp`: EVO IR Interpreter
    pub itp: Option<Rc<RefCell<Interpreter>>>,
}


impl CPUState {

    // =================== IRCtx.const ===================== //
    // Set Constants
    /// Base of Addr: 0x04000000
    pub const DEFAULT_BASE_ADDR: usize = 0x04000000;
    /// Mem size: default 4MB = 4 * 1024 * 1024
    pub const DEFAULT_MEM_SIZE: usize = 4 * 1024 * 1024;
    /// Stack Mem size: default 1MB = 1 * 1024 * 1024
    pub const DEFAULT_STACK_SIZE: usize = 1 * 1024 * 1024;

    // =================== IRCtx.ctl ======================= //

    /// Init a `CPUState`
    /// default:
    /// 1. `base_addr`: 0x04000000
    /// 2. `mem_size`: 4 * 1024 * 1024
    /// 3. `stack_size`: 1 * 1024 * 1024
    pub fn init(src_arch: &'static Arch, ir_arch: &'static Arch, base_addr: Option<usize>, mem_size: Option<usize>, stack_size: Option<usize>) -> Self {
        let base_addr = base_addr.unwrap_or(Self::DEFAULT_BASE_ADDR);
        let mem_size = mem_size.unwrap_or(Self::DEFAULT_MEM_SIZE);
        let stack_size = stack_size.unwrap_or(Self::DEFAULT_STACK_SIZE);
        let cpu = Self {
            src_arch,
            ir_arch,
            proc: CPUProcess::init(src_arch, base_addr, mem_size, stack_size),
            itp: Interpreter::itp_pool_init(ir_arch),
        };
        cpu
    }

    /// excute by Interpreter
    pub fn execute(&self, insn: &Instruction) {
        if let Some(itp) = &self.itp {
            itp.borrow_mut().execute(self, insn);
        }
    }

    // =================== IRCtx.get ======================= //


    /// Get Arch string
    pub fn to_string (&self) -> String {
        format!("{}", self.src_arch)
    }

    /// Get CPUConfig string
    pub fn info(&self) -> String {
        format!("src: {}, ir: {}", self.src_arch, self.ir_arch.name)
    }

    /// Get src Name
    pub const fn src_name(&self) -> &'static str {
        self.src_arch.name
    }

    // =================== IRCtx.is ======================== //

    /// Check if `CPUState` is init
    pub fn is_init(&self) -> bool {
        RegFile::reg_pool_num(self.ir_arch) != 0 && Instruction::insn_pool_size() != 0
    }

    /// Clear Temp Insn Pool
    pub fn pool_clr(&self) {
        // 1. Check is init
        if !self.is_init() {
            log_warning!("CPUState not init");
            return
        }

        RegFile::reg_pool_clr(self.ir_arch);
        Instruction::insn_pool_clr();
        Interpreter::func_pool_clr();
    }

    /// Info of Pools
    pub fn pool_info(&self) -> String{
        let str = format!(
            "{}\n{}\n{}",
            RegFile::reg_pool_info(self.ir_arch),
            Instruction::insn_pool_info(),
            Interpreter::func_pool_info()
        );
        str
    }

    // =================== IRCtx.process =================== //

    /// Info of reg pool
    pub fn reg_info(&self, start: usize, num: i32) -> String {
        self.proc.borrow().reg_info(start, num)
    }

    /// Set reg by index
    pub fn set_reg(&self, index: usize, value: Value) {
        self.proc.borrow_mut().set_reg(index, value);
    }

    /// Get reg by index
    pub fn get_reg(&self, index: usize) -> Value {
        self.proc.borrow().get_reg(index)
    }

    /// Set reg by name
    pub fn set_nreg(&self, name: &'static str, value: Value) {
        self.proc.borrow_mut().set_nreg(name, value);
    }

    /// Get reg val by name
    pub fn get_nreg(&self, name: &'static str) -> Value {
        self.proc.borrow().get_nreg(name)
    }

    /// Get pc
    pub fn get_pc(&self) -> Value {
        self.proc.borrow().get_pc()
    }

    /// Set pc
    pub fn set_pc(&self, value: Value) {
        self.proc.borrow_mut().set_pc(value);
    }

    /// Read Mem
    pub fn mem_read(&self, index: usize, num: usize) -> Value {
        self.proc.borrow().mem_read(index, num)
    }

    /// Write Mem
    pub fn mem_write(&self, index: usize, value: Value) {
        self.proc.borrow_mut().mem_write(index, value);
    }

    /// Get status
    pub fn status(&self) -> CPUThreadStatus {
        self.proc.borrow().status()
    }

    // =================== IRCtx.decode ==================== //
    

}


impl Default for CPUState {
    /// Set default function for `CPUState`.
    fn default() -> Self {
        Self::init(&RISCV32_ARCH, &EVO_ARCH, None, None, None)
    }
}






// ============================================================================== //
//                                Unit Tests
// ============================================================================== //

#[cfg(test)]
mod cpu_test {

    use super::*;
    use crate::core::{mem::CPUThread, op::Operand};

    #[test]
    fn insn_info() {
        CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);

        let insn1 = Instruction::insn_pool_nget("sub").borrow().clone();
        println!("{}", insn1.info());
        println!("{}", insn1.bin(0, -1, true));

        let insn2 = Instruction::apply(
            "sub", vec![
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x31").borrow().clone(), 
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x0").borrow().clone(), 
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x8").borrow().clone()
            ]
        );
        println!("{}", insn2.bin(0, -1, true));
        let insn3 = Instruction::apply(
            "srl", vec![
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x31").borrow().clone(), 
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x30").borrow().clone(), 
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x7").borrow().clone()
            ]
        );
        println!("{}", insn3.bin(0, -1, true));
        println!("{}", insn3);

        let insn4 = Instruction::apply(
            "addi", vec![
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x31").borrow().clone(),
                RegFile::reg_poolr_nget(&RISCV32_ARCH, "x30").borrow().clone(),
                Operand::imm(Value::u12(2457)),
            ]
        );
        println!("{}", insn4.bin(0, -1, true));
        println!("{}", insn4);
    }

    #[test]
    fn insn_decode() {
        CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);
        let insn1 = Instruction::decode(&RISCV32_ARCH, Value::from_string("0B01000000 10000000 00001111 10110011"));  // sub x32, x0, x8
        println!("{}", insn1);
        let insn2 = Instruction::decode(&RISCV32_ARCH, Value::from_string("0B00000000 00001000 00110000 00110011"));  // sltu x0, x16, x0
        println!("{}", insn2);
        println!("{}", insn2.arch);

    }

    #[test]
    fn mem_info() {
        let cpu = CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);
        let p0 = cpu.proc;
        println!("{}", CPUProcess::pool_info_tbl());

        let t0 = p0.borrow_mut().cur_thread.clone();
        t0.borrow_mut().stack_push(Value::array(vec![Value::u64(1), Value::u64(2)]));
        let t1 = p0.borrow_mut().fork_thread();
        t1.borrow_mut().stack_push(Value::array(vec![Value::u64(3), Value::u64(4)]));
        let t2 = p0.borrow_mut().fork_thread();
        t2.borrow_mut().stack_push(Value::array(vec![Value::u64(5), Value::u64(6)]));
        let t3 = p0.borrow_mut().fork_thread();
        t3.borrow_mut().stack_push(Value::array(vec![Value::u64(7), Value::u64(8)]));
        let t4 = p0.borrow_mut().fork_thread();
        t4.borrow_mut().stack_push(Value::array(vec![Value::u64(9), Value::u64(10)]));
        t4.borrow_mut().status = CPUThreadStatus::Unknown;
        
        p0.borrow_mut().set_thread(3);
        let p1 = CPUProcess::init(&RISCV32_ARCH, 0, 1024, 1024);
        p1.borrow_mut().stack_push(Value::array(vec![Value::u64(1), Value::u64(2), Value::u64(3), Value::u64(4)]));

        let t5 = p0.borrow_mut().fork_thread();
        t5.borrow_mut().stack_push(Value::array(vec![Value::u64(11), Value::u64(12)]));

        println!("{}", CPUThread::pool_info_tbl());
        println!("{}", CPUProcess::pool_info_tbl());


    }

    #[test]
    fn cpu_info() {
        // println!("{}", CPUState::info());
        let cpu = CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);

        // Check pool info
        println!("{}", cpu.pool_info());

        let p0 = cpu.proc.borrow().clone();
        // Check process info
        println!("{}", p0.info());
        p0.new_thread();
        println!("{}", cpu.proc.borrow().info());

        p0.set_reg(3, Value::u32(23));
        println!("{}", cpu.proc.borrow().reg_info(0, 4));
        println!("{}", p0.get_reg(3));

        p0.mem_write(13, Value::i32(-65535));
        println!("{}", p0.mem_read(13, 2));
    }

}