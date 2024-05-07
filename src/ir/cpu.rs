//! `evo::ir::cpu` : IR Context
//! 
//!

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, ArchMode};
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

    const ARCH: Arch;
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
    /// `proc`: Process Handle
    pub proc: Rc<RefCell<CPUProcess>>,
    /// `itp`: Interpreter
    pub itp: Option<Rc<RefCell<Interpreter>>>,
}

impl CPUConfig for CPUState {

    // =================== IRCtx.const ===================== //

    const ARCH: Arch = Arch::def(ArchKind::RISCV, ArchMode::BIT32, 32);
    // Set Constants
    const BASE_ADDR: usize = 0x04000000;
    const MEM_SIZE: usize = 4 * 1024 * 1024;
    const STACK_SIZE: usize = 1 * 1024 * 1024;

}

impl CPUState {

    // =================== IRCtx.ctl ======================= //

    /// Init a `CPUState`
    pub fn init() -> Self {
        let cpu = Self {
            proc: CPUProcess::init(Self::name()),
            itp: Self::pool_init()
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
    pub fn to_string () -> String {
        format!("{}", Self::name())
    }

    /// Get CPUConfig string
    pub fn info() -> String {
        format!("{}", Self::ARCH)
    }

    /// Get Name
    pub const fn name() -> &'static str {
        Self::ARCH.name
    }

    /// Get reg num
    pub const fn reg_num() -> usize {
        Self::ARCH.reg_num
    }

    /// Get Arch Rc
    pub const fn arch() -> Arch {
        Self::ARCH
    }

    // =================== IRCtx.is ======================== //

    /// Check if `CPUState` is init
    pub fn is_init() -> bool {
        Instruction::reg_pool_size() != 0 && Instruction::insn_pool_size() != 0
    }

    /// Check if is_32
    pub const fn is_32() -> bool {
        Self::ARCH.mode.is_32bit()
    }

    /// Check if is_64
    pub const fn is_64() -> bool {
        Self::ARCH.mode.is_64bit()
    }

    // =================== IRCtx.pool ====================== //

    /// Insn temp and Reg and Interpreter Pool Init
    pub fn pool_init() -> Option<Rc<RefCell<Interpreter>>> {
        // 1. Check is init
        if Self::is_init() {
            log_warning!("CPUState is already init");
            return None;
        }
        // 2. Init regs pool
        Instruction::reg("x0", Value::bit(5, 0));
        Instruction::reg("x1", Value::bit(5, 1));
        Instruction::reg("x2", Value::bit(5, 2));
        Instruction::reg("x3", Value::bit(5, 3));
        Instruction::reg("x4", Value::bit(5, 4));
        Instruction::reg("x5", Value::bit(5, 5));
        Instruction::reg("x6", Value::bit(5, 6));
        Instruction::reg("x7", Value::bit(5, 7));
        Instruction::reg("x8", Value::bit(5, 8));
        Instruction::reg("x9", Value::bit(5, 9));
        Instruction::reg("x10", Value::bit(5, 10));
        Instruction::reg("x11", Value::bit(5, 11));
        Instruction::reg("x12", Value::bit(5, 12));
        Instruction::reg("x13", Value::bit(5, 13));
        Instruction::reg("x14", Value::bit(5, 14));
        Instruction::reg("x15", Value::bit(5, 15));
        Instruction::reg("x16", Value::bit(5, 16));
        Instruction::reg("x17", Value::bit(5, 17));
        Instruction::reg("x18", Value::bit(5, 18));
        Instruction::reg("x19", Value::bit(5, 19));
        Instruction::reg("x20", Value::bit(5, 20));
        Instruction::reg("x21", Value::bit(5, 21));
        Instruction::reg("x22", Value::bit(5, 22));
        Instruction::reg("x23", Value::bit(5, 23));
        Instruction::reg("x24", Value::bit(5, 24));
        Instruction::reg("x25", Value::bit(5, 25));
        Instruction::reg("x26", Value::bit(5, 26));
        Instruction::reg("x27", Value::bit(5, 27));
        Instruction::reg("x28", Value::bit(5, 28));
        Instruction::reg("x29", Value::bit(5, 29));
        Instruction::reg("x30", Value::bit(5, 30));
        Instruction::reg("x31", Value::bit(5, 31));

        // 3. Init insns & insns interpreter
        let itp = Interpreter::init(Self::arch());
        // RISCV Instruction Format:                                           32|31  25|24 20|19 15|  |11  7|6    0|
        // Type: R                                [rd, rs1, rs2]                 |  f7  | rs2 | rs1 |f3|  rd |  op  |
        itp.def_insn("add" , vec![1, 1, 1], "R", "0B0000000. ........ .000.... .0110011", 
            |cpu, insn| {
                // ======== rd = rs1 + rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Add rs1 and rs2
                let res = rs1.wrapping_add(rs2);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("sub" , vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
            |cpu, insn| {
                // ======== rd = rs1 - rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Sub rs1 and rs2
                let res = rs1.wrapping_sub(rs2);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("or"  , vec![1, 1, 1], "R", "0B0000000. ........ .110.... .0110011",
            |cpu, insn| {
                // ======== rd = rs1 | rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Or rs1 and rs2
                let res = rs1 | rs2;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("xor" , vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
            |cpu, insn| {
                // ======== rd = rs1 ^ rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Xor rs1 and rs2
                let res = rs1 ^ rs2;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("and" , vec![1, 1, 1], "R", "0B0000000. ........ .111.... .0110011",
            |cpu, insn| {
                // ======== rd = rs1 & rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. And rs1 and rs2
                let res = rs1 & rs2;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("sll" , vec![1, 1, 1], "R", "0B0000000. ........ .001.... .0110011",
            |cpu, insn| {
                // ====== rd = rs1 << rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Sll rs1 and rs2
                let res = rs1.wrapping_shl(rs2);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("srl" , vec![1, 1, 1], "R", "0B0000000. ........ .101.... .0110011",
            |cpu, insn| {
                // ====== rd = rs1 >> rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Srl rs1 and rs2
                let res = rs1.wrapping_shr(rs2);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("sra" , vec![1, 1, 1], "R", "0B0100000. ........ .101.... .0110011",
            |cpu, insn| {
                // ====== rd = rs1 >> rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Sra rs1 and rs2: Shift Right Arith
                let res = rs1.wrapping_shr(rs2);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("slt" , vec![1, 1, 1], "R", "0B0000000. ........ .010.... .0110011",
            |cpu, insn| {
                // ======== rd = rs1 < rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Slt rs1 and rs2
                let res = if rs1 < rs2 { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("sltu", vec![1, 1, 1], "R", "0B0000000. ........ .011.... .0110011",
            |cpu, insn| {
                // ======== rd = rs1 < rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Sltu rs1 and rs2
                let res = if rs1 < rs2 { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        // Type: I                                [rd, rs1, imm]                 |    imm     | rs1 |f3|  rd |  op  |
        itp.def_insn("addi", vec![1, 1, 0], "I", "0B........ ........ .000.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 + imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Add rs1 and imm
                let res = rs1 + imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("xori", vec![1, 1, 0], "I", "0B........ ........ .100.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 ^ imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Xor rs1 and imm
                let res = rs1 ^ imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("ori" , vec![1, 1, 0], "I", "0B........ ........ .110.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 | imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Or rs1 and imm
                let res = rs1 | imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("andi", vec![1, 1, 0], "I", "0B........ ........ .111.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 & imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. And rs1 and imm
                let res = rs1 & imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("slli", vec![1, 1, 0], "I", "0B0000000. ........ .001.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 << imm[0:4] ======= //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm[0:4](u32)
                let imm = (insn.imm_i() & 0x1F) as u32;
                // 3. Sll rs1 and imm
                let res = rs1 << imm;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("srli", vec![1, 1, 0], "I", "0B0000000. ........ .101.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 >> imm[0:4] ======= //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm[0:4](u32)
                let imm = (insn.imm_i() & 0x1F) as u32;
                // 3. Sll rs1 and imm
                let res = rs1 >> imm;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("srai", vec![1, 1, 0], "I", "0B0100000. ........ .101.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 >> imm[0:4] ======= //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm[0:4](u32)
                let imm = (insn.imm_i() & 0x1F) as u32;
                // 3. Sll rs1 and imm
                let res = rs1 >> imm;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(res));
            }
        );
        itp.def_insn("slti", vec![1, 1, 0], "I", "0B........ ........ .010.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 < imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Slt rs1 and imm
                let res = if rs1 < imm { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("sltiu",vec![1, 1, 0], "I", "0B........ ........ .011.... .0010011",
            |cpu, insn| {
                // ======== rd = rs1 < imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm(u32)
                let imm = insn.imm_i() as u32;
                // 3. Slt rs1 and imm
                let res = if rs1 < imm { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(res));
            }
        );
        itp.def_insn("lb", vec![1, 1, 0], "I", "0B........ ........ .000.... .0000011",
            |cpu, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Byte
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_byte(0);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
            }
        );
        itp.def_insn("lh", vec![1, 1, 0], "I", "0B........ ........ .001.... .0000011", 
            |cpu, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Half
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_half(0);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
            }
        );
        itp.def_insn("lw", vec![1, 1, 0], "I", "0B........ ........ .010.... .0000011",
            |cpu, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Word
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_word(0);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
            }
        );
        itp.def_insn("lbu", vec![1, 1, 0], "I", "0B........ ........ .100.... .0000011",
            |cpu, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Byte
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_byte(0);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(val as u32));
            }
        );
        itp.def_insn("lhu", vec![1, 1, 0], "I", "0B........ ........ .101.... .0000011",
            |cpu, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Half
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_half(0);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, Value::u32(val as u32));
            }
        );
        itp.def_insn("jalr", vec![1, 1, 0], "I", "0B........ ........ .000.... .1100111",
            |cpu, insn| {
                // ======== rd = pc + 4; pc = rs1 + imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(proc0.get_pc().get_i32(0) + 4));
                // 4. Set pc(i32)
                proc0.set_pc(Value::i32(rs1 + imm));
            }
        );
        itp.def_insn("ecall", vec![], "I", "0B00000000 0000.... .000.... .1110111",
            |cpu, insn| {
                // ======== ecall ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // System will do next part according to register `a7`(x17)
                proc0.set_status(CPUThreadStatus::Blocked);
            }
        );
        itp.def_insn("ebreak", vec![], "I", "0B00000000 0001.... .000.... .1110111",
            |cpu, insn| {
                // ======== ebreak ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // Debugger will do next part according to register `a7`(x17)
                proc0.set_status(CPUThreadStatus::Blocked);
            }
        );
        // Type:S 
        itp.def_insn("sb", vec![1, 1, 0], "S", "0B........ ........ .000.... .0100011",
            |cpu, insn| {
                // ======== [rs1 + imm] = rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_s() as i32;
                // 3. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_byte(0) as i8;
                // 4. Write Mem: Byte
                proc0.write_mem((rs1 + imm) as usize, Value::i8(rs2));
            }
        );
        itp.def_insn("sh", vec![1, 1, 0], "S", "0B........ ........ .001.... .0100011",
            |cpu, insn| {
                // ======== [rs1 + imm] = rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_s() as i32;
                // 3. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_half(0) as i16;
                // 4. Write Mem: Half
                proc0.write_mem((rs1 + imm) as usize, Value::i16(rs2));
            }
        );
        itp.def_insn("sw", vec![1, 1, 0], "S", "0B........ ........ .010.... .0100011",
            |cpu, insn| {
                // ======== [rs1 + imm] = rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_s() as i32;
                // 3. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_word(0) as i32;
                // 4. Write Mem: Word
                proc0.write_mem((rs1 + imm) as usize, Value::i32(rs2));
            }
        );
        // Type: B
        itp.def_insn("beq", vec![1, 1, 0], "B", "0B........ ........ .000.... .1100011",
            |cpu, insn| {
                // ======== if(rs1 == rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 == rs2 {
                    proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        itp.def_insn("bne", vec![1, 1, 0], "B", "0B........ ........ .001.... .1100011",
            |cpu, insn| {
                // ======== if(rs1 != rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 != rs2 {
                    proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        itp.def_insn("blt", vec![1, 1, 0], "B", "0B........ ........ .100.... .1100011",
            |cpu, insn| {
                // ======== if(rs1 < rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 < rs2 {
                    proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        itp.def_insn("bge", vec![1, 1, 0], "B", "0B........ ........ .101.... .1100011",
            |cpu, insn| {
                // ======== if(rs1 >= rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 >= rs2 {
                    proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        itp.def_insn("bltu", vec![1, 1, 0], "B", "0B........ ........ .110.... .1100011",
            |cpu, insn| {
                // ======== if(rs1 < rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 < rs2 {
                    proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        itp.def_insn("bgeu", vec![1, 1, 0], "B", "0B........ ........ .111.... .1100011",
            |cpu, insn| {
                // ======== if(rs1 >= rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 >= rs2 {
                    proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        // Type: U
        itp.def_insn("lui", vec![1, 0], "U", "0B........ ........ ........ .0110111",
            |cpu, insn| {
                // ======== rd = imm << 12 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get imm(i32)
                let imm = insn.imm_u() as i32;
                // 2. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(imm << 12));
            }
        );
        itp.def_insn("auipc", vec![1, 0], "U", "0B........ ........ ........ .0010111",
            |cpu, insn| {
                // ======== rd = pc + imm << 12 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get imm(i32)
                let imm = insn.imm_u() as i32;
                // 2. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(proc0.get_pc().get_i32(0) + imm << 12));
            }
        );
        // Type: J
        itp.def_insn("jal", vec![1, 0], "J", "0B........ ........ ........ .1101111",
            |cpu, insn| {
                // ======== rd = pc + 4; pc = pc + imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = cpu.proc.borrow().clone();
                // 1. Get imm(i32)
                let imm = insn.imm_j() as i32;
                // 2. Get pc(i32)
                let pc = proc0.get_pc().get_i32(0);
                // 3. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, Value::i32(pc + 4));
                // 4. Set PC
                proc0.set_pc(Value::i32(pc + imm));
            }
        );


        Some(Rc::new(RefCell::new(itp)))
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
        Interpreter::pool_clr();
    }

    /// Info of Pools
    pub fn pool_info() -> String{
        let str = format!(
            "{}\n{}\n{}",
            Instruction::reg_pool_info(),
            Instruction::insn_pool_info(),
            Interpreter::pool_info()
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
        Self::init()
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
        CPUState::init();

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
        CPUState::init();
        let insn1 = Instruction::decode(Value::from_string("0B01000000 10000000 00001111 10110011"));  // sub x32, x0, x8
        println!("{}", insn1);
        let insn2 = Instruction::decode(Value::from_string("0B00000000 00001000 00110000 00110011"));  // sltu x0, x16, x0
        println!("{}", insn2);
        println!("{}", insn2.arch);

    }

    #[test]
    fn mem_info() {
        let cpu = CPUState::init();
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
        let p1 = CPUProcess::init("test");
        p1.borrow_mut().stack_push(Value::array(vec![Value::u64(1), Value::u64(2), Value::u64(3), Value::u64(4)]));

        let t5 = p0.borrow_mut().fork_thread();
        t5.borrow_mut().stack_push(Value::array(vec![Value::u64(11), Value::u64(12)]));

        println!("{}", CPUThread::pool_info_tbl());
        println!("{}", CPUProcess::pool_info_tbl());


    }

    #[test]
    fn cpu_info() {
        // println!("{}", CPUState::info());
        let cpu = CPUState::init();

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
        let cpu = CPUState::init();
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