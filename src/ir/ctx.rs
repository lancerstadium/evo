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
use crate::ir::val::IRValue;
use crate::ir::op::IRInsn;
use crate::ir::mem::IRProcess;
use crate::ir::itp::IRInterpreter;

use super::op::IROperand;


// ============================================================================== //
//                                ctx::ArchInfo
// ============================================================================== //


/// `ArchInfo`: Config information of the architecture
pub trait ArchInfo {


    // ====================== Const ====================== //

    /// Arch name: like "evo"
    const NAME: &'static str;
    /// Number of bytes in a byte: *1*, 2, 4
    const BYTE_SIZE: usize;
    /// Number of bytes in a addr(ptr/reg.size: 0x00 ~ 2^ADDR_SIZE): 8, 16, *32*, 64
    const ADDR_SIZE: usize;
    /// Number of bytes in a word(interger): 8, 16, **32**, 64
    const WORD_SIZE: usize;
    /// Number of bytes in a (float): **32**, 64
    const FLOAT_SIZE: usize;
    /// Base of Addr: 0x04000000
    const BASE_ADDR: usize;
    /// Mem size: default 4MB = 4 * 1024 * 1024
    const MEM_SIZE: usize;
    /// Stack Mem size: default 1MB = 1 * 1024 * 1024
    const STACK_SIZE: usize;
    /// Number of Registers: 8, 16, **32**, 64
    const REG_NUM: usize;

}


// ============================================================================== //
//                              ctx::IRContext
// ============================================================================== //


/// `IRContext`: Context of the `evo-ir` architecture
#[derive(Clone, PartialEq)]
pub struct IRContext {
    /// `proc`: Process Handle
    proc: Rc<RefCell<IRProcess>>,
    /// `itp`: Interpreter
    pub itp: Option<Rc<RefCell<IRInterpreter>>>,
}

impl ArchInfo for IRContext {

    // =================== IRCtx.const ===================== //

    // Set Constants
    const NAME: &'static str = "evo32";
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
            proc: Rc::new(RefCell::new(IRProcess::default())),
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
        IRInsn::reg("x0", IRValue::u5(0));
        IRInsn::reg("x1", IRValue::u5(1));
        IRInsn::reg("x2", IRValue::u5(2));
        IRInsn::reg("x3", IRValue::u5(3));
        IRInsn::reg("x4", IRValue::u5(4));
        IRInsn::reg("x5", IRValue::u5(5));
        IRInsn::reg("x6", IRValue::u5(6));
        IRInsn::reg("x7", IRValue::u5(7));
        IRInsn::reg("x8", IRValue::u5(8));
        IRInsn::reg("x9", IRValue::u5(9));
        IRInsn::reg("x10", IRValue::u5(10));
        IRInsn::reg("x11", IRValue::u5(11));
        IRInsn::reg("x12", IRValue::u5(12));
        IRInsn::reg("x13", IRValue::u5(13));
        IRInsn::reg("x14", IRValue::u5(14));
        IRInsn::reg("x15", IRValue::u5(15));
        IRInsn::reg("x16", IRValue::u5(16));
        IRInsn::reg("x17", IRValue::u5(17));
        IRInsn::reg("x18", IRValue::u5(18));
        IRInsn::reg("x19", IRValue::u5(19));
        IRInsn::reg("x20", IRValue::u5(20));
        IRInsn::reg("x21", IRValue::u5(21));
        IRInsn::reg("x22", IRValue::u5(22));
        IRInsn::reg("x23", IRValue::u5(23));
        IRInsn::reg("x24", IRValue::u5(24));
        IRInsn::reg("x25", IRValue::u5(25));
        IRInsn::reg("x26", IRValue::u5(26));
        IRInsn::reg("x27", IRValue::u5(27));
        IRInsn::reg("x28", IRValue::u5(28));
        IRInsn::reg("x29", IRValue::u5(29));
        IRInsn::reg("x30", IRValue::u5(30));
        IRInsn::reg("x31", IRValue::u5(31));

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
        IRInterpreter::def_insn("or"  , vec![1, 1, 1], "R", "0B0000000. ........ .111.... .0110011",
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
                let res = rs1 << rs2;
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
                let res = rs1 >> rs2;
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
                let res = rs1 >> rs2;
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
                // 4. Set rd(u32)
                proc0.set_reg(insn.rd() as usize, IRValue::u32(res));
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

            }
        );
        IRInterpreter::def_insn("ori" , vec![1, 1, 0], "I", "0B........ ........ .110.... .0010011",
            |ctx, insn| {
                
            }
        );
        IRInterpreter::def_insn("andi", vec![1, 1, 0], "I", "0B........ ........ .111.... .0010011",
            |ctx, insn| {

            }
        );
        IRInterpreter::def_insn("slli", vec![1, 1, 0], "I", "0B0000000. ........ .001.... .0010011",
            |ctx, insn| {

            }
        );
        IRInterpreter::def_insn("srli", vec![1, 1, 0], "I", "0B0000000. ........ .101.... .0010011",
            |ctx, insn| {

            }
        );
        IRInterpreter::def_insn("srai", vec![1, 1, 0], "I", "0B0100000. ........ .101.... .0010011",
            |ctx, insn| {

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
            }
        );
        IRInterpreter::def_insn("sltiu",vec![1, 1, 0], "I", "0B........ ........ .011.... .0010011",
            |ctx, insn| {

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
    pub fn reg_info(&self) -> String {
        self.proc.borrow().reg_info()
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
    use crate::ir::op::IROperand;

    #[test]
    fn insn_info() {
        IRContext::init();

        let insn1 = IRInsn::insn_pool_nget("sub").borrow().clone();
        println!("{}", insn1.info());
        println!("{}", insn1.bin(0, -1, true));

        let insn2 = IRInsn::apply(
            "sub", vec![
                IRInsn::reg_pool_nget("x31").borrow().clone(), 
                IRInsn::reg_pool_nget("x2").borrow().clone(), 
                IRInsn::reg_pool_nget("x8").borrow().clone()
            ]
        );
        println!("{}", insn2.bin(0, -1, true));
        println!("{}", insn2);

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
        println!("{}", ctx.proc.borrow().reg_info());
        println!("{}", p0.get_reg(3));
    }


    #[test]
    fn ctx_excute() {
        let ctx = IRContext::init();
        let insn = IRInsn::apply("add", vec![
            IRInsn::reg_pool_nget("x0").borrow().clone(),
            IRInsn::reg_pool_nget("x1").borrow().clone(),
            IRInsn::reg_pool_nget("x2").borrow().clone()
        ]);

        ctx.set_nreg("x1", IRValue::u32(2));
        ctx.set_nreg("x2", IRValue::u32(3));
        println!("{}", ctx.reg_info());
        ctx.execute(&insn);
        println!("{}", ctx.reg_info());
    }

}