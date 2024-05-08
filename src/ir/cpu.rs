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
use crate::ir::val::Value;
use crate::ir::insn::Instruction;
use crate::ir::mem::CPUProcess;
use crate::ir::itp::Interpreter;
use crate::ir::mem::CPUThreadStatus;



// ============================================================================== //
//                                cpu::CPUConfig
// ============================================================================== //


/// `CPUConfig`: Config information of the CPU architecture
pub trait CPUConfig {

    // ====================== Const ====================== //

    /// IR Architecture
    const IR_ARCH : Arch;
    /// Base of Addr: 0x04000000
    const BASE_ADDR: usize;
    /// Mem size: default 4MB = 4 * 1024 * 1024
    const MEM_SIZE: usize;
    /// Stack Mem size: default 1MB = 1 * 1024 * 1024
    const STACK_SIZE: usize;

}


// ============================================================================== //
//                              cpu::CPUState
// ============================================================================== //


/// `CPUState`: Context of the `evo-ir` architecture
#[derive(Clone, PartialEq)]
pub struct CPUState {
    pub src_arch: &'static Arch,
    /// `proc`: Process Handle
    pub proc: Rc<RefCell<CPUProcess>>,
    /// `itp`: EVO IR Interpreter
    pub itp: Option<Rc<RefCell<Interpreter>>>,
}

impl CPUConfig for CPUState {

    // =================== IRCtx.const ===================== //
    // Set Constants
    const IR_ARCH : Arch = EVO_ARCH;
    const BASE_ADDR: usize = 0x04000000;
    const MEM_SIZE: usize = 4 * 1024 * 1024;
    const STACK_SIZE: usize = 1 * 1024 * 1024;

}

impl CPUState {

    // =================== IRCtx.ctl ======================= //

    /// Init a `CPUState`
    pub fn init(src_arch: &'static Arch) -> Self {
        let cpu = Self {
            src_arch,
            proc: CPUProcess::init(src_arch),
            itp: Interpreter::itp_pool_init(&Self::IR_ARCH),
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

    /// Get ir name
    pub const fn ir_name() -> &'static str {
        Self::IR_ARCH.name
    }

    /// Get ir reg num
    pub const fn ir_reg_num() -> usize {
        Self::IR_ARCH.reg_num
    }

    /// Get ir Arch
    pub const fn ir_arch() -> &'static Arch {
        &Self::IR_ARCH
    }

    /// Get guest/source Arch Rc
    pub const fn src_arch(&self) -> &'static Arch {
        self.src_arch
    }
    /// Get Arch string
    pub fn to_string (&self) -> String {
        format!("{}", self.src_arch)
    }

    /// Get CPUConfig string
    pub fn info(&self) -> String {
        format!("src: {}, ir: {}", self.src_arch, Self::ir_name())
    }

    /// Get src Name
    pub const fn src_name(&self) -> &'static str {
        self.src_arch.name
    }

    // =================== IRCtx.is ======================== //

    /// Check if `CPUState` is init
    pub fn is_init() -> bool {
        Instruction::reg_pool_size() != 0 && Instruction::insn_pool_size() != 0
    }

    /// Check if is_32
    pub const fn is_ir_32() -> bool {
        Self::IR_ARCH.mode.is_32bit()
    }

    /// Check if is_64
    pub const fn is_ir_64() -> bool {
        Self::IR_ARCH.mode.is_64bit()
    }

    /// Clear Temp Insn Pool
    pub fn pool_clr() {
        // 1. Check is init
        if !Self::is_init() {
            log_warning!("CPUState not init");
            return
        }

        Instruction::reg_pool_clr();
        Instruction::insn_pool_clr();
        Interpreter::func_pool_clr();
    }

    /// Info of Pools
    pub fn pool_info() -> String{
        let str = format!(
            "{}\n{}\n{}",
            Instruction::reg_pool_info(),
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

    /// Get reg by name
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
    pub fn read_mem(&self, index: usize, num: usize) -> Value {
        self.proc.borrow().read_mem(index, num)
    }

    /// Write Mem
    pub fn write_mem(&self, index: usize, value: Value) {
        self.proc.borrow_mut().write_mem(index, value);
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
        Self::init(&RISCV32_ARCH)
    }
}






// ============================================================================== //
//                                Unit Tests
// ============================================================================== //

#[cfg(test)]
mod cpu_test {

    use super::*;
    use crate::ir::{mem::CPUThread, op::Operand};

    #[test]
    fn insn_info() {
        CPUState::init(&RISCV32_ARCH);

        let insn1 = Instruction::insn_pool_nget("sub").borrow().clone();
        println!("{}", insn1.info());
        println!("{}", insn1.bin(0, -1, true));

        let insn2 = Instruction::apply(
            "sub", vec![
                Instruction::reg_pool_nget("x31").borrow().clone(), 
                Instruction::reg_pool_nget("x0").borrow().clone(), 
                Instruction::reg_pool_nget("x8").borrow().clone()
            ]
        );
        println!("{}", insn2.bin(0, -1, true));
        let insn3 = Instruction::apply(
            "srl", vec![
                Instruction::reg_pool_nget("x31").borrow().clone(), 
                Instruction::reg_pool_nget("x30").borrow().clone(), 
                Instruction::reg_pool_nget("x7").borrow().clone()
            ]
        );
        println!("{}", insn3.bin(0, -1, true));
        println!("{}", insn3);

        let insn4 = Instruction::apply(
            "addi", vec![
                Instruction::reg_pool_nget("x31").borrow().clone(),
                Instruction::reg_pool_nget("x30").borrow().clone(),
                Operand::imm(Value::u12(2457)),
            ]
        );
        println!("{}", insn4.bin(0, -1, true));
        println!("{}", insn4);
    }

    #[test]
    fn insn_decode() {
        CPUState::init(&RISCV32_ARCH);
        let insn1 = Instruction::decode(Value::from_string("0B01000000 10000000 00001111 10110011"));  // sub x32, x0, x8
        println!("{}", insn1);
        let insn2 = Instruction::decode(Value::from_string("0B00000000 00001000 00110000 00110011"));  // sltu x0, x16, x0
        println!("{}", insn2);
        println!("{}", insn2.arch);

    }

    #[test]
    fn mem_info() {
        let cpu = CPUState::init(&RISCV32_ARCH);
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
        let p1 = CPUProcess::init(&RISCV32_ARCH);
        p1.borrow_mut().stack_push(Value::array(vec![Value::u64(1), Value::u64(2), Value::u64(3), Value::u64(4)]));

        let t5 = p0.borrow_mut().fork_thread();
        t5.borrow_mut().stack_push(Value::array(vec![Value::u64(11), Value::u64(12)]));

        println!("{}", CPUThread::pool_info_tbl());
        println!("{}", CPUProcess::pool_info_tbl());


    }

    #[test]
    fn cpu_info() {
        // println!("{}", CPUState::info());
        let cpu = CPUState::init(&RISCV32_ARCH);

        // Check pool info
        println!("{}", CPUState::pool_info());

        let p0 = cpu.proc.borrow().clone();
        // Check process info
        println!("{}", p0.info());
        p0.new_thread();
        println!("{}", cpu.proc.borrow().info());

        p0.set_reg(3, Value::u32(23));
        println!("{}", cpu.proc.borrow().reg_info(0, 4));
        println!("{}", p0.get_reg(3));

        p0.write_mem(13, Value::i32(-65535));
        println!("{}", p0.read_mem(13, 2));
    }


    #[test]
    fn cpu_excute() {
        let cpu = CPUState::init(&RISCV32_ARCH);
        cpu.set_nreg("x1", Value::i32(3));
        cpu.set_nreg("x2", Value::i32(5));
        cpu.set_nreg("x3", Value::i32(-32));
        cpu.write_mem(26, Value::i32(0x1ffff));

        // R-Type Insns Test
        let insn1 = Instruction::from_string("add x0, x1, x2");
        let insn2 = Instruction::from_string("sub x0, x1, x2");
        let insn3 = Instruction::from_string("or x0, x1, x2");
        let insn4 = Instruction::from_string("xor x0, x1, x2");
        let insn5 = Instruction::from_string("and x0, x1, x2");
        let insn6 = Instruction::from_string("sltu x0, x1, x2");
        let insn7 = Instruction::from_string("srl x0, x1, x3");
        let insn8 = Instruction::from_string("sra x0, x1, x3");
        let insn9 = Instruction::from_string("sll x0, x1, x3");
        let insn10 = Instruction::from_string("slt x0, x1, x2");

        // I-Type Insns Test
        let insn11 = Instruction::from_string("addi x0, x1, 2457");
        let insn12 = Instruction::from_string("andi x0, x1, 2457");
        let insn13 = Instruction::from_string("ori x0, x1, 2457");
        let insn14 = Instruction::from_string("xori x0, x1, 2457");
        let insn15 = Instruction::from_string("slti x0, x1, 2");
        let insn16 = Instruction::from_string("sltiu x0, x1, 2");

        let insn17 = Instruction::from_string("lb x0, x1, 23");
        let insn18 = Instruction::from_string("lh x0, x1, 23");
        let insn19 = Instruction::from_string("lw x0, x1, 23");
        let insn20 = Instruction::from_string("lbu x0, x1, 23");
        let insn21 = Instruction::from_string("lhu x0, x1, 23");

        let insn22 = Instruction::from_string("sb x0, x1, 23");
        let insn23 = Instruction::from_string("sh x0, x1, 23");
        let insn24 = Instruction::from_string("sw x0, x1, 23");
        
        let insn34 = Instruction::from_string("jalr x0, x1, 23");
        let insn35 = Instruction::from_string("ecall");
        let insn36 = Instruction::from_string("ebreak");

        // B-Type Insns Test
        let insn25 = Instruction::from_string("beq x0, x1, 23");
        let insn26 = Instruction::from_string("bne x0, x1, 23");
        let insn27 = Instruction::from_string("blt x0, x1, 23");
        let insn28 = Instruction::from_string("bge x0, x1, 23");
        let insn29 = Instruction::from_string("bltu x0, x1, 23");
        let insn30 = Instruction::from_string("bgeu x0, x1, 23");

        // U-Type Insns Test
        let insn31 = Instruction::from_string("lui x0, 255");
        let insn32 = Instruction::from_string("auipc x0, 255");
        // J-Type Insns Test
        let insn33 = Instruction::from_string("jal x0, 23");

        cpu.execute(&insn1);
        println!("{:<50} -> x0 = {}", insn1.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn2);
        println!("{:<50} -> x0 = {}", insn2.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn3);
        println!("{:<50} -> x0 = {}", insn3.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn4);
        println!("{:<50} -> x0 = {}", insn4.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn5);
        println!("{:<50} -> x0 = {}", insn5.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn6);
        println!("{:<50} -> x0 = {}", insn6.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn7);
        println!("{:<50} -> x0 = {}", insn7.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn8);
        println!("{:<50} -> x0 = {}", insn8.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn9);
        println!("{:<50} -> x0 = {}", insn9.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn10);
        println!("{:<50} -> x0 = {}", insn10.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn11);
        println!("{:<50} -> x0 = {}", insn11.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn12);
        println!("{:<50} -> x0 = {}", insn12.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn13);
        println!("{:<50} -> x0 = {}", insn13.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn14);
        println!("{:<50} -> x0 = {}", insn14.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn15);
        println!("{:<50} -> x0 = {}", insn15.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn16);
        println!("{:<50} -> x0 = {}", insn16.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn17);
        println!("{:<50} -> x0 = {}", insn17.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn18);
        println!("{:<50} -> x0 = {}", insn18.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn19);
        println!("{:<50} -> x0 = {}", insn19.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn20);
        println!("{:<50} -> x0 = {}", insn20.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn21);
        println!("{:<50} -> x0 = {}", insn21.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.set_nreg("x0", Value::i32(56));
        cpu.execute(&insn22);
        println!("{:<50} -> mem = {}", insn22.to_string(), cpu.read_mem(26, 1).bin(0, 1, false));
        cpu.set_nreg("x0", Value::i32(732));
        cpu.execute(&insn23);
        println!("{:<50} -> mem = {}", insn23.to_string(), cpu.read_mem(26, 1).bin(0, 2, false));
        cpu.set_nreg("x0", Value::i32(-8739));
        cpu.execute(&insn24);
        println!("{:<50} -> mem = {}", insn24.to_string(), cpu.read_mem(26, 1).bin(0, 4, false));

        cpu.execute(&insn25);
        println!("{:<50} -> pc = {}", insn25.to_string(), cpu.get_pc());
        cpu.execute(&insn26);
        println!("{:<50} -> pc = {}", insn26.to_string(), cpu.get_pc());
        cpu.execute(&insn27);
        println!("{:<50} -> pc = {}", insn27.to_string(), cpu.get_pc());
        cpu.execute(&insn28);
        println!("{:<50} -> pc = {}", insn28.to_string(), cpu.get_pc());
        cpu.execute(&insn29);
        println!("{:<50} -> pc = {}", insn29.to_string(), cpu.get_pc());
        cpu.execute(&insn30);
        println!("{:<50} -> pc = {}", insn30.to_string(), cpu.get_pc());

        cpu.execute(&insn31);
        println!("{:<50} -> pc = {}, x0 = {}", insn31.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn32);
        println!("{:<50} -> pc = {}, x0 = {}", insn32.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn33);
        println!("{:<50} -> pc = {}, x0 = {}", insn33.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn34);
        println!("{:<50} -> pc = {}, x0 = {}", insn34.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn35);
        println!("{:<50} -> status = {}", insn35.to_string(), cpu.status());
        cpu.execute(&insn36);
        println!("{:<50} -> status = {}", insn36.to_string(), cpu.status());
    }

}