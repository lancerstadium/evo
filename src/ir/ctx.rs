//! `evo::ir::ctx` : IR Context
//! 
//!

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::ArchInfo;
use crate::ir::val::IRValue;
use crate::ir::op::IRInsn;
use crate::ir::mem::IRProcess;
use crate::ir::itp::IRInterpreter;
use crate::ir::mem::IRThreadStatus;



// ============================================================================== //
//                              ctx::IRContext
// ============================================================================== //


/// `IRContext`: Context of the `evo-ir` architecture
#[derive(Clone, PartialEq)]
pub struct IRContext {
    /// `proc`: Process Handle
    pub proc: Rc<RefCell<IRProcess>>,
    /// `itp`: Interpreter
    pub itp: Option<Rc<RefCell<IRInterpreter>>>,
}

impl ArchInfo for IRContext {

    // =================== IRCtx.const ===================== //

    // Set Constants
    const NAME: &'static str = "riscv32";
    const BYTE_SIZE: usize = 1;
    const ADDR_SIZE: usize = 32;
    const WORD_SIZE: usize = 32;
    const FLOAT_SIZE: usize = 32;
    const BASE_ADDR: usize = 0x04000000;
    const MEM_SIZE: usize = 4 * 1024 * 1024;
    const STACK_SIZE: usize = 1 * 1024 * 1024;
    const REG_NUM: usize = 32;

}

impl IRContext {

    // =================== IRCtx.ctl ======================= //

    /// Init a `IRContext`
    pub fn init() -> Self {
        let ctx = Self {
            proc: IRProcess::init(Self::name()),
            itp: Self::pool_init()
        };
        ctx
    }

    /// excute by Interpreter
    pub fn execute(&self, insn: &IRInsn) {
        if let Some(itp) = &self.itp {
            itp.borrow_mut().execute(self, insn);
        }
    }

    // =================== IRCtx.get ======================= //

    /// Get Arch string
    pub fn to_string () -> String {
        format!("{}", Self::NAME)
    }

    /// Get ArchInfo string
    pub fn info() -> String {
        format!("Arch Info: \n- Name: {}\n- Byte Size: {}\n- Addr Size: {}\n- Word Size: {}\n- Float Size: {}\n- Base Addr: 0x{:x}\n- Mem Size: {}\n- Reg Num: {}\n", 
            Self::NAME, Self::BYTE_SIZE, Self::ADDR_SIZE, Self::WORD_SIZE, Self::FLOAT_SIZE, Self::BASE_ADDR, Self::STACK_SIZE, Self::REG_NUM)
    }

    /// Get Name
    pub fn name() -> &'static str {
        Self::NAME
    }

    /// Get status
    pub fn status(&self) -> IRThreadStatus {
        self.proc.borrow().status()
    }

    // =================== IRCtx.is ======================== //

    /// Check if `IRContext` is init
    pub fn is_init() -> bool {
        IRInsn::reg_pool_size() != 0 && IRInsn::insn_pool_size() != 0
    }

    /// Check if is_32
    pub fn is_32() -> bool {
        Self::ADDR_SIZE == 32
    }

    /// Check if is_64
    pub fn is_64() -> bool {
        Self::ADDR_SIZE == 64
    }

    // =================== IRCtx.pool ====================== //

    /// Insn temp and Reg and Interpreter Pool Init
    pub fn pool_init() -> Option<Rc<RefCell<IRInterpreter>>> {
        // 1. Check is init
        if Self::is_init() {
            log_warning!("IRContext is already init");
            return None;
        }
        // 2. Init regs pool
        IRInsn::reg("x0", IRValue::bit(5, 0));
        IRInsn::reg("x1", IRValue::bit(5, 1));
        IRInsn::reg("x2", IRValue::bit(5, 2));
        IRInsn::reg("x3", IRValue::bit(5, 3));
        IRInsn::reg("x4", IRValue::bit(5, 4));
        IRInsn::reg("x5", IRValue::bit(5, 5));
        IRInsn::reg("x6", IRValue::bit(5, 6));
        IRInsn::reg("x7", IRValue::bit(5, 7));
        IRInsn::reg("x8", IRValue::bit(5, 8));
        IRInsn::reg("x9", IRValue::bit(5, 9));
        IRInsn::reg("x10", IRValue::bit(5, 10));
        IRInsn::reg("x11", IRValue::bit(5, 11));
        IRInsn::reg("x12", IRValue::bit(5, 12));
        IRInsn::reg("x13", IRValue::bit(5, 13));
        IRInsn::reg("x14", IRValue::bit(5, 14));
        IRInsn::reg("x15", IRValue::bit(5, 15));
        IRInsn::reg("x16", IRValue::bit(5, 16));
        IRInsn::reg("x17", IRValue::bit(5, 17));
        IRInsn::reg("x18", IRValue::bit(5, 18));
        IRInsn::reg("x19", IRValue::bit(5, 19));
        IRInsn::reg("x20", IRValue::bit(5, 20));
        IRInsn::reg("x21", IRValue::bit(5, 21));
        IRInsn::reg("x22", IRValue::bit(5, 22));
        IRInsn::reg("x23", IRValue::bit(5, 23));
        IRInsn::reg("x24", IRValue::bit(5, 24));
        IRInsn::reg("x25", IRValue::bit(5, 25));
        IRInsn::reg("x26", IRValue::bit(5, 26));
        IRInsn::reg("x27", IRValue::bit(5, 27));
        IRInsn::reg("x28", IRValue::bit(5, 28));
        IRInsn::reg("x29", IRValue::bit(5, 29));
        IRInsn::reg("x30", IRValue::bit(5, 30));
        IRInsn::reg("x31", IRValue::bit(5, 31));

        // 3. Init insns & insns interpreter
        let itp = IRInterpreter::init();
        // RISCV Instruction Format:                                           32|31  25|24 20|19 15|  |11  7|6    0|
        // Type: R                                [rd, rs1, rs2]                 |  f7  | rs2 | rs1 |f3|  rd |  op  |
        IRInterpreter::def_insn("add" , vec![1, 1, 1], "R", "0B0000000. ........ .000.... .0110011", 
            |ctx, insn| {
                // ======== rd = rs1 + rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Add rs1 and rs2
                let res = rs1.wrapping_add(rs2);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("sub" , vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
            |ctx, insn| {
                // ======== rd = rs1 - rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Sub rs1 and rs2
                let res = rs1.wrapping_sub(rs2);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("or"  , vec![1, 1, 1], "R", "0B0000000. ........ .110.... .0110011",
            |ctx, insn| {
                // ======== rd = rs1 | rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Or rs1 and rs2
                let res = rs1 | rs2;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("xor" , vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
            |ctx, insn| {
                // ======== rd = rs1 ^ rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Xor rs1 and rs2
                let res = rs1 ^ rs2;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("and" , vec![1, 1, 1], "R", "0B0000000. ........ .111.... .0110011",
            |ctx, insn| {
                // ======== rd = rs1 & rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. And rs1 and rs2
                let res = rs1 & rs2;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("sll" , vec![1, 1, 1], "R", "0B0000000. ........ .001.... .0110011",
            |ctx, insn| {
                // ====== rd = rs1 << rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Sll rs1 and rs2
                let res = rs1.wrapping_shl(rs2);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("srl" , vec![1, 1, 1], "R", "0B0000000. ........ .101.... .0110011",
            |ctx, insn| {
                // ====== rd = rs1 >> rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Srl rs1 and rs2
                let res = rs1.wrapping_shr(rs2);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("sra" , vec![1, 1, 1], "R", "0B0100000. ........ .101.... .0110011",
            |ctx, insn| {
                // ====== rd = rs1 >> rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Sra rs1 and rs2: Shift Right Arith
                let res = rs1.wrapping_shr(rs2);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("slt" , vec![1, 1, 1], "R", "0B0000000. ........ .010.... .0110011",
            |ctx, insn| {
                // ======== rd = rs1 < rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Slt rs1 and rs2
                let res = if rs1 < rs2 { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("sltu", vec![1, 1, 1], "R", "0B0000000. ........ .011.... .0110011",
            |ctx, insn| {
                // ======== rd = rs1 < rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Sltu rs1 and rs2
                let res = if rs1 < rs2 { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        // Type: I                                [rd, rs1, imm]                 |    imm     | rs1 |f3|  rd |  op  |
        IRInterpreter::def_insn("addi", vec![1, 1, 0], "I", "0B........ ........ .000.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 + imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Add rs1 and imm
                let res = rs1 + imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("xori", vec![1, 1, 0], "I", "0B........ ........ .100.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 ^ imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Xor rs1 and imm
                let res = rs1 ^ imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("ori" , vec![1, 1, 0], "I", "0B........ ........ .110.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 | imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Or rs1 and imm
                let res = rs1 | imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("andi", vec![1, 1, 0], "I", "0B........ ........ .111.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 & imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. And rs1 and imm
                let res = rs1 & imm;
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("slli", vec![1, 1, 0], "I", "0B0000000. ........ .001.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 << imm[0:4] ======= //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm[0:4](u32)
                let imm = (insn.imm_i() & 0x1F) as u32;
                // 3. Sll rs1 and imm
                let res = rs1 << imm;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("srli", vec![1, 1, 0], "I", "0B0000000. ........ .101.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 >> imm[0:4] ======= //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm[0:4](u32)
                let imm = (insn.imm_i() & 0x1F) as u32;
                // 3. Sll rs1 and imm
                let res = rs1 >> imm;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("srai", vec![1, 1, 0], "I", "0B0100000. ........ .101.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 >> imm[0:4] ======= //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm[0:4](u32)
                let imm = (insn.imm_i() & 0x1F) as u32;
                // 3. Sll rs1 and imm
                let res = rs1 >> imm;
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
            }
        );
        IRInterpreter::def_insn("slti", vec![1, 1, 0], "I", "0B........ ........ .010.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 < imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Slt rs1 and imm
                let res = if rs1 < imm { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("sltiu",vec![1, 1, 0], "I", "0B........ ........ .011.... .0010011",
            |ctx, insn| {
                // ======== rd = rs1 < imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get imm(u32)
                let imm = insn.imm_i() as u32;
                // 3. Slt rs1 and imm
                let res = if rs1 < imm { 1 } else { 0 };
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(res));
            }
        );
        IRInterpreter::def_insn("lb", vec![1, 1, 0], "I", "0B........ ........ .000.... .0000011",
            |ctx, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Byte
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_byte(0);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(val as i32));
            }
        );
        IRInterpreter::def_insn("lh", vec![1, 1, 0], "I", "0B........ ........ .001.... .0000011", 
            |ctx, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Half
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_half(0);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(val as i32));
            }
        );
        IRInterpreter::def_insn("lw", vec![1, 1, 0], "I", "0B........ ........ .010.... .0000011",
            |ctx, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Word
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_word(0);
                // 4. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(val as i32));
            }
        );
        IRInterpreter::def_insn("lbu", vec![1, 1, 0], "I", "0B........ ........ .100.... .0000011",
            |ctx, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Byte
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_byte(0);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(val as u32));
            }
        );
        IRInterpreter::def_insn("lhu", vec![1, 1, 0], "I", "0B........ ........ .101.... .0000011",
            |ctx, insn| {
                // ======== rd = [rs1 + imm] ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Read Mem: Half
                let val = proc0.read_mem((rs1 + imm) as usize, 1).get_half(0);
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(val as u32));
            }
        );
        IRInterpreter::def_insn("jalr", vec![1, 1, 0], "I", "0B........ ........ .000.... .1100111",
            |ctx, insn| {
                // ======== rd = pc + 4; pc = rs1 + imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_i() as i32;
                // 3. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(proc0.get_pc().get_i32(0) + 4));
                // 4. Set pc(i32)
                proc0.set_pc(IRValue::i32(rs1 + imm));
            }
        );
        IRInterpreter::def_insn("ecall", vec![], "I", "0B00000000 0000.... .000.... .1110111",
            |ctx, insn| {
                // ======== ecall ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // System will do next part according to register `a7`(x17)
                proc0.set_status(IRThreadStatus::Blocked);
            }
        );
        IRInterpreter::def_insn("ebreak", vec![], "I", "0B00000000 0001.... .000.... .1110111",
            |ctx, insn| {
                // ======== ebreak ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // Debugger will do next part according to register `a7`(x17)
                proc0.set_status(IRThreadStatus::Blocked);
            }
        );
        // Type:S 
        IRInterpreter::def_insn("sb", vec![1, 1, 0], "S", "0B........ ........ .000.... .0100011",
            |ctx, insn| {
                // ======== [rs1 + imm] = rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_s() as i32;
                // 3. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_byte(0) as i8;
                // 4. Write Mem: Byte
                proc0.write_mem((rs1 + imm) as usize, IRValue::i8(rs2));
            }
        );
        IRInterpreter::def_insn("sh", vec![1, 1, 0], "S", "0B........ ........ .001.... .0100011",
            |ctx, insn| {
                // ======== [rs1 + imm] = rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_s() as i32;
                // 3. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_half(0) as i16;
                // 4. Write Mem: Half
                proc0.write_mem((rs1 + imm) as usize, IRValue::i16(rs2));
            }
        );
        IRInterpreter::def_insn("sw", vec![1, 1, 0], "S", "0B........ ........ .010.... .0100011",
            |ctx, insn| {
                // ======== [rs1 + imm] = rs2 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get imm(i32)
                let imm = insn.imm_s() as i32;
                // 3. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_word(0) as i32;
                // 4. Write Mem: Word
                proc0.write_mem((rs1 + imm) as usize, IRValue::i32(rs2));
            }
        );
        // Type: B
        IRInterpreter::def_insn("beq", vec![1, 1, 0], "B", "0B........ ........ .000.... .1100011",
            |ctx, insn| {
                // ======== if(rs1 == rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 == rs2 {
                    proc0.set_pc(IRValue::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        IRInterpreter::def_insn("bne", vec![1, 1, 0], "B", "0B........ ........ .001.... .1100011",
            |ctx, insn| {
                // ======== if(rs1 != rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 != rs2 {
                    proc0.set_pc(IRValue::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        IRInterpreter::def_insn("blt", vec![1, 1, 0], "B", "0B........ ........ .100.... .1100011",
            |ctx, insn| {
                // ======== if(rs1 < rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 < rs2 {
                    proc0.set_pc(IRValue::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        IRInterpreter::def_insn("bge", vec![1, 1, 0], "B", "0B........ ........ .101.... .1100011",
            |ctx, insn| {
                // ======== if(rs1 >= rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(i32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
                // 2. Get rs2(i32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 >= rs2 {
                    proc0.set_pc(IRValue::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        IRInterpreter::def_insn("bltu", vec![1, 1, 0], "B", "0B........ ........ .110.... .1100011",
            |ctx, insn| {
                // ======== if(rs1 < rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 < rs2 {
                    proc0.set_pc(IRValue::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        IRInterpreter::def_insn("bgeu", vec![1, 1, 0], "B", "0B........ ........ .111.... .1100011",
            |ctx, insn| {
                // ======== if(rs1 >= rs2) pc += imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get rs1(u32)
                let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
                // 2. Get rs2(u32)
                let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
                // 3. Get imm(i32)
                let imm = insn.imm_b() as i32;
                // 4. Set PC
                if rs1 >= rs2 {
                    proc0.set_pc(IRValue::i32(proc0.get_pc().get_i32(0) + imm));
                }
            }
        );
        // Type: U
        IRInterpreter::def_insn("lui", vec![1, 0], "U", "0B........ ........ ........ .0110111",
            |ctx, insn| {
                // ======== rd = imm << 12 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get imm(i32)
                let imm = insn.imm_u() as i32;
                // 2. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(imm << 12));
            }
        );
        IRInterpreter::def_insn("auipc", vec![1, 0], "U", "0B........ ........ ........ .0010111",
            |ctx, insn| {
                // ======== rd = pc + imm << 12 ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get imm(i32)
                let imm = insn.imm_u() as i32;
                // 2. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(proc0.get_pc().get_i32(0) + imm << 12));
            }
        );
        // Type: J
        IRInterpreter::def_insn("jal", vec![1, 0], "J", "0B........ ........ ........ .1101111",
            |ctx, insn| {
                // ======== rd = pc + 4; pc = pc + imm ======== //
                if !insn.is_applied {
                    log_warning!("Insn not applied: {}", insn);
                    return;
                }
                let proc0 = ctx.proc.borrow().clone();
                // 1. Get imm(i32)
                let imm = insn.imm_j() as i32;
                // 2. Get pc(i32)
                let pc = proc0.get_pc().get_i32(0);
                // 3. Set rd(i32)
                proc0.set_reg(insn.rd() as usize, IRValue::i32(pc + 4));
                // 4. Set PC
                proc0.set_pc(IRValue::i32(pc + imm));
            }
        );


        Some(Rc::new(RefCell::new(itp)))
    }

    /// Clear Temp Insn Pool
    pub fn pool_clr() {
        // 1. Check is init
        if !Self::is_init() {
            log_warning!("IRContext not init");
            return
        }

        IRInsn::reg_pool_clr();
        IRInsn::insn_pool_clr();
        IRInterpreter::pool_clr();
    }

    /// Info of Pools
    pub fn pool_info() -> String{
        let str = format!(
            "{}\n{}\n{}",
            IRInsn::reg_pool_info(),
            IRInsn::insn_pool_info(),
            IRInterpreter::pool_info()
        );
        str
    }

    // =================== IRCtx.process =================== //

    /// Info of reg pool
    pub fn reg_info(&self, start: usize, num: i32) -> String {
        self.proc.borrow().reg_info(start, num)
    }

    /// Set reg by index
    pub fn set_reg(&self, index: usize, value: IRValue) {
        self.proc.borrow_mut().set_reg(index, value);
    }

    /// Get reg by index
    pub fn get_reg(&self, index: usize) -> IRValue {
        self.proc.borrow().get_reg(index)
    }

    /// Set reg by name
    pub fn set_nreg(&self, name: &'static str, value: IRValue) {
        self.proc.borrow_mut().set_nreg(name, value);
    }

    /// Get reg by name
    pub fn get_nreg(&self, name: &'static str) -> IRValue {
        self.proc.borrow().get_nreg(name)
    }

    /// Get pc
    pub fn get_pc(&self) -> IRValue {
        self.proc.borrow().get_pc()
    }

    /// Set pc
    pub fn set_pc(&self, value: IRValue) {
        self.proc.borrow_mut().set_pc(value);
    }

    /// Read Mem
    pub fn read_mem(&self, index: usize, num: usize) -> IRValue {
        self.proc.borrow().read_mem(index, num)
    }

    /// Write Mem
    pub fn write_mem(&self, index: usize, value: IRValue) {
        self.proc.borrow_mut().write_mem(index, value);
    }

    // =================== IRCtx.decode ==================== //
    

}


impl Default for IRContext {
    /// Set default function for `IRContext`.
    fn default() -> Self {
        Self::init()
    }
}






// ============================================================================== //
//                                Unit Tests
// ============================================================================== //

#[cfg(test)]
mod ctx_test {

    use super::*;
    use crate::ir::{mem::IRThread, op::IROperand};

    #[test]
    fn insn_info() {
        IRContext::init();

        let insn1 = IRInsn::insn_pool_nget("sub").borrow().clone();
        println!("{}", insn1.info());
        println!("{}", insn1.bin(0, -1, true));

        let insn2 = IRInsn::apply(
            "sub", vec![
                IRInsn::reg_pool_nget("x31").borrow().clone(), 
                IRInsn::reg_pool_nget("x0").borrow().clone(), 
                IRInsn::reg_pool_nget("x8").borrow().clone()
            ]
        );
        println!("{}", insn2.bin(0, -1, true));
        let insn3 = IRInsn::apply(
            "srl", vec![
                IRInsn::reg_pool_nget("x31").borrow().clone(), 
                IRInsn::reg_pool_nget("x30").borrow().clone(), 
                IRInsn::reg_pool_nget("x7").borrow().clone()
            ]
        );
        println!("{}", insn3.bin(0, -1, true));
        println!("{}", insn3);

        let insn4 = IRInsn::apply(
            "addi", vec![
                IRInsn::reg_pool_nget("x31").borrow().clone(),
                IRInsn::reg_pool_nget("x30").borrow().clone(),
                IROperand::imm(IRValue::u12(2457)),
            ]
        );
        println!("{}", insn4.bin(0, -1, true));
        println!("{}", insn4);
    }

    #[test]
    fn insn_decode() {
        IRContext::init();
        let insn1 = IRInsn::decode(IRValue::from_string("0B01000000 10000000 00001111 10110011"));  // sub x32, x0, x8
        println!("{}", insn1);
        let insn2 = IRInsn::decode(IRValue::from_string("0B00000000 00001000 00110000 00110011"));  // sltu x0, x16, x0
        println!("{}", insn2);

    }

    #[test]
    fn mem_info() {
        let ctx = IRContext::init();
        let p0 = ctx.proc;
        println!("{}", IRProcess::pool_info_tbl());

        let t0 = p0.borrow_mut().cur_thread.clone();
        t0.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(1), IRValue::u64(2)]));
        let t1 = p0.borrow_mut().fork_thread();
        t1.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(3), IRValue::u64(4)]));
        let t2 = p0.borrow_mut().fork_thread();
        t2.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(5), IRValue::u64(6)]));
        let t3 = p0.borrow_mut().fork_thread();
        t3.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(7), IRValue::u64(8)]));
        let t4 = p0.borrow_mut().fork_thread();
        t4.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(9), IRValue::u64(10)]));
        t4.borrow_mut().status = IRThreadStatus::Unknown;
        
        p0.borrow_mut().set_thread(3);
        let p1 = IRProcess::init("test");
        p1.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(1), IRValue::u64(2), IRValue::u64(3), IRValue::u64(4)]));

        let t5 = p0.borrow_mut().fork_thread();
        t5.borrow_mut().stack_push(IRValue::array(vec![IRValue::u64(11), IRValue::u64(12)]));

        println!("{}", IRThread::pool_info_tbl());
        println!("{}", IRProcess::pool_info_tbl());


    }

    #[test]
    fn ctx_info() {
        // println!("{}", IRContext::info());
        let ctx = IRContext::init();

        // Check pool info
        println!("{}", IRContext::pool_info());

        let p0 = ctx.proc.borrow().clone();
        // Check process info
        println!("{}", p0.info());
        p0.new_thread();
        println!("{}", ctx.proc.borrow().info());

        p0.set_reg(3, IRValue::u32(23));
        println!("{}", ctx.proc.borrow().reg_info(0, 4));
        println!("{}", p0.get_reg(3));

        p0.write_mem(13, IRValue::i32(-65535));
        println!("{}", p0.read_mem(13, 2));
    }


    #[test]
    fn ctx_excute() {
        let ctx = IRContext::init();
        ctx.set_nreg("x1", IRValue::i32(3));
        ctx.set_nreg("x2", IRValue::i32(5));
        ctx.set_nreg("x3", IRValue::i32(-32));
        ctx.write_mem(26, IRValue::i32(0x1ffff));

        // R-Type Insns Test
        let insn1 = IRInsn::from_string("add x0, x1, x2");
        let insn2 = IRInsn::from_string("sub x0, x1, x2");
        let insn3 = IRInsn::from_string("or x0, x1, x2");
        let insn4 = IRInsn::from_string("xor x0, x1, x2");
        let insn5 = IRInsn::from_string("and x0, x1, x2");
        let insn6 = IRInsn::from_string("sltu x0, x1, x2");
        let insn7 = IRInsn::from_string("srl x0, x1, x3");
        let insn8 = IRInsn::from_string("sra x0, x1, x3");
        let insn9 = IRInsn::from_string("sll x0, x1, x3");
        let insn10 = IRInsn::from_string("slt x0, x1, x2");

        // I-Type Insns Test
        let insn11 = IRInsn::from_string("addi x0, x1, 2457");
        let insn12 = IRInsn::from_string("andi x0, x1, 2457");
        let insn13 = IRInsn::from_string("ori x0, x1, 2457");
        let insn14 = IRInsn::from_string("xori x0, x1, 2457");
        let insn15 = IRInsn::from_string("slti x0, x1, 2");
        let insn16 = IRInsn::from_string("sltiu x0, x1, 2");

        let insn17 = IRInsn::from_string("lb x0, x1, 23");
        let insn18 = IRInsn::from_string("lh x0, x1, 23");
        let insn19 = IRInsn::from_string("lw x0, x1, 23");
        let insn20 = IRInsn::from_string("lbu x0, x1, 23");
        let insn21 = IRInsn::from_string("lhu x0, x1, 23");

        let insn22 = IRInsn::from_string("sb x0, x1, 23");
        let insn23 = IRInsn::from_string("sh x0, x1, 23");
        let insn24 = IRInsn::from_string("sw x0, x1, 23");
        
        let insn34 = IRInsn::from_string("jalr x0, x1, 23");
        let insn35 = IRInsn::from_string("ecall");
        let insn36 = IRInsn::from_string("ebreak");

        // B-Type Insns Test
        let insn25 = IRInsn::from_string("beq x0, x1, 23");
        let insn26 = IRInsn::from_string("bne x0, x1, 23");
        let insn27 = IRInsn::from_string("blt x0, x1, 23");
        let insn28 = IRInsn::from_string("bge x0, x1, 23");
        let insn29 = IRInsn::from_string("bltu x0, x1, 23");
        let insn30 = IRInsn::from_string("bgeu x0, x1, 23");

        // U-Type Insns Test
        let insn31 = IRInsn::from_string("lui x0, 255");
        let insn32 = IRInsn::from_string("auipc x0, 255");
        // J-Type Insns Test
        let insn33 = IRInsn::from_string("jal x0, 23");

        ctx.execute(&insn1);
        println!("{:<50} -> x0 = {}", insn1.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn2);
        println!("{:<50} -> x0 = {}", insn2.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn3);
        println!("{:<50} -> x0 = {}", insn3.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn4);
        println!("{:<50} -> x0 = {}", insn4.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn5);
        println!("{:<50} -> x0 = {}", insn5.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn6);
        println!("{:<50} -> x0 = {}", insn6.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn7);
        println!("{:<50} -> x0 = {}", insn7.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn8);
        println!("{:<50} -> x0 = {}", insn8.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn9);
        println!("{:<50} -> x0 = {}", insn9.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn10);
        println!("{:<50} -> x0 = {}", insn10.to_string(), ctx.get_nreg("x0").get_i32(0));

        ctx.execute(&insn11);
        println!("{:<50} -> x0 = {}", insn11.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn12);
        println!("{:<50} -> x0 = {}", insn12.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn13);
        println!("{:<50} -> x0 = {}", insn13.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn14);
        println!("{:<50} -> x0 = {}", insn14.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn15);
        println!("{:<50} -> x0 = {}", insn15.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn16);
        println!("{:<50} -> x0 = {}", insn16.to_string(), ctx.get_nreg("x0").get_i32(0));

        ctx.execute(&insn17);
        println!("{:<50} -> x0 = {}", insn17.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn18);
        println!("{:<50} -> x0 = {}", insn18.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn19);
        println!("{:<50} -> x0 = {}", insn19.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn20);
        println!("{:<50} -> x0 = {}", insn20.to_string(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn21);
        println!("{:<50} -> x0 = {}", insn21.to_string(), ctx.get_nreg("x0").get_i32(0));

        ctx.set_nreg("x0", IRValue::i32(56));
        ctx.execute(&insn22);
        println!("{:<50} -> mem = {}", insn22.to_string(), ctx.read_mem(26, 1).bin(0, 1, false));
        ctx.set_nreg("x0", IRValue::i32(732));
        ctx.execute(&insn23);
        println!("{:<50} -> mem = {}", insn23.to_string(), ctx.read_mem(26, 1).bin(0, 2, false));
        ctx.set_nreg("x0", IRValue::i32(-8739));
        ctx.execute(&insn24);
        println!("{:<50} -> mem = {}", insn24.to_string(), ctx.read_mem(26, 1).bin(0, 4, false));

        ctx.execute(&insn25);
        println!("{:<50} -> pc = {}", insn25.to_string(), ctx.get_pc());
        ctx.execute(&insn26);
        println!("{:<50} -> pc = {}", insn26.to_string(), ctx.get_pc());
        ctx.execute(&insn27);
        println!("{:<50} -> pc = {}", insn27.to_string(), ctx.get_pc());
        ctx.execute(&insn28);
        println!("{:<50} -> pc = {}", insn28.to_string(), ctx.get_pc());
        ctx.execute(&insn29);
        println!("{:<50} -> pc = {}", insn29.to_string(), ctx.get_pc());
        ctx.execute(&insn30);
        println!("{:<50} -> pc = {}", insn30.to_string(), ctx.get_pc());

        ctx.execute(&insn31);
        println!("{:<50} -> pc = {}, x0 = {}", insn31.to_string(), ctx.get_pc(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn32);
        println!("{:<50} -> pc = {}, x0 = {}", insn32.to_string(), ctx.get_pc(), ctx.get_nreg("x0").get_i32(0));

        ctx.execute(&insn33);
        println!("{:<50} -> pc = {}, x0 = {}", insn33.to_string(), ctx.get_pc(), ctx.get_nreg("x0").get_i32(0));
        ctx.execute(&insn34);
        println!("{:<50} -> pc = {}, x0 = {}", insn34.to_string(), ctx.get_pc(), ctx.get_nreg("x0").get_i32(0));

        ctx.execute(&insn35);
        println!("{:<50} -> status = {}", insn35.to_string(), ctx.status());
        ctx.execute(&insn36);
        println!("{:<50} -> status = {}", insn36.to_string(), ctx.status());
    }

}