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
use crate::ir::op::{IROperand, IRInsn};
use crate::ir::mem::IRProcess;


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
    /// Mem size: default 64MB = 4 * 1024 * 1024
    const MEM_SIZE: usize;
    /// Number of Registers: 8, 16, **32**, 64
    const REG_NUM: usize;

}


// ============================================================================== //
//                              ctx::IRContext
// ============================================================================== //


/// `IRContext`: Context of the `evo-ir` architecture
/// 

#[derive(Debug, Clone, PartialEq)]
pub struct IRContext {

    proc: Rc<RefCell<IRProcess>>,
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
    const REG_NUM: usize = 32;

}

impl IRContext {

    // =================== IRCtx.ctr ======================= //

    /// Init a `IRContext`
    pub fn init() -> Self {
        let ctx = Self {
            proc: Rc::new(RefCell::new(IRProcess::default())),
        };
        Self::pool_init();
        ctx
    }

    // =================== IRCtx.get ======================= //

    /// Get Arch string
    pub fn to_string () -> String {
        format!("{}", Self::NAME)
    }

    /// Get ArchInfo string
    pub fn info() -> String {
        format!("Arch Info: \n- Name: {}\n- Byte Size: {}\n- Addr Size: {}\n- Word Size: {}\n- Float Size: {}\n- Base Addr: 0x{:x}\n- Mem Size: {}\n- Reg Num: {}\n", 
            Self::NAME, Self::BYTE_SIZE, Self::ADDR_SIZE, Self::WORD_SIZE, Self::FLOAT_SIZE, Self::BASE_ADDR, Self::MEM_SIZE, Self::REG_NUM)
    }

    /// Get Name
    pub fn name() -> &'static str {
        Self::NAME
    }

    // =================== IRCtx.is ======================== //

    /// Check if `IRContext` is init
    pub fn is_init() -> bool {
        IROperand::pool_size() != 0 && IRInsn::pool_size() != 0
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

    /// Insn temp and Reg Pool Init
    pub fn pool_init() {
        // 1. Check is init
        if Self::is_init() {
            log_warning!("IRContext is already init");
            return
        }
        // 2. Init regs pool
        IROperand::reg("x0", IRValue::u5(0));
        IROperand::reg("x1", IRValue::u5(1));
        IROperand::reg("x2", IRValue::u5(2));
        IROperand::reg("x3", IRValue::u5(3));
        IROperand::reg("x4", IRValue::u5(4));
        IROperand::reg("x5", IRValue::u5(5));
        IROperand::reg("x6", IRValue::u5(6));
        IROperand::reg("x7", IRValue::u5(7));
        IROperand::reg("x8", IRValue::u5(8));
        IROperand::reg("x9", IRValue::u5(9));
        IROperand::reg("x10", IRValue::u5(10));
        IROperand::reg("x11", IRValue::u5(11));
        IROperand::reg("x12", IRValue::u5(12));
        IROperand::reg("x13", IRValue::u5(13));
        IROperand::reg("x14", IRValue::u5(14));
        IROperand::reg("x15", IRValue::u5(15));
        IROperand::reg("x16", IRValue::u5(16));
        IROperand::reg("x17", IRValue::u5(17));
        IROperand::reg("x18", IRValue::u5(18));
        IROperand::reg("x19", IRValue::u5(19));
        IROperand::reg("x20", IRValue::u5(20));
        IROperand::reg("x21", IRValue::u5(21));
        IROperand::reg("x22", IRValue::u5(22));
        IROperand::reg("x23", IRValue::u5(23));
        IROperand::reg("x24", IRValue::u5(24));
        IROperand::reg("x25", IRValue::u5(25));
        IROperand::reg("x26", IRValue::u5(26));
        IROperand::reg("x27", IRValue::u5(27));
        IROperand::reg("x28", IRValue::u5(28));
        IROperand::reg("x29", IRValue::u5(29));
        IROperand::reg("x30", IRValue::u5(30));
        IROperand::reg("x31", IRValue::u5(31));

        // 3. Init insns
        // RISCV Instruction Format:                               32|31  25|24 20|19 15|  |11  7|6    0|
        // Type: R                    [rd, rs1, rs2]                 |  f7  | rs2 | rs1 |f3|  rd |  op  |
        IRInsn::def("add" , vec![1, 1, 1], "R", "0b0000000. ........ .000.... .0110011");
        IRInsn::def("sub" , vec![1, 1, 1], "R", "0b0100000. ........ .000.... .0110011");
        IRInsn::def("or"  , vec![1, 1, 1], "R", "0b0000000. ........ .111.... .0110011");
        IRInsn::def("xor" , vec![1, 1, 1], "R", "0b0000000. ........ .100.... .0110011");
        IRInsn::def("sll" , vec![1, 1, 1], "R", "0b0000000. ........ .001.... .0110011");
        IRInsn::def("srl" , vec![1, 1, 1], "R", "0b0000000. ........ .101.... .0110011");
        IRInsn::def("sra" , vec![1, 1, 1], "R", "0b0100000. ........ .101.... .0110011");
        IRInsn::def("slt" , vec![1, 1, 1], "R", "0b0000000. ........ .010.... .0110011");
        IRInsn::def("sltu", vec![1, 1, 1], "R", "0b0000000. ........ .011.... .0110011");
        // Type: I                    [rd, rs1, imm]                 |    imm     | rs1 |f3|  rd |  op  |
        IRInsn::def("addi", vec![1, 1, 0], "I", "0b0000000. ........ .000.... .0010011");
    }


    /// Clear Temp Insn Pool
    pub fn pool_clr() {
        // 1. Check is init
        if !Self::is_init() {
            log_warning!("IRContext not init");
            return
        }

        IROperand::pool_clr();
        IRInsn::pool_clr();
    }

    /// Info of Pools
    pub fn pool_info() -> String{
        let str = format!(
            "{}\n{}",
            IROperand::pool_info(),
            IRInsn::pool_info()
        );
        str
    }


    // =================== IRCtx.process =================== //

    
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


    #[test]
    fn insn_info() {
        IRContext::init();
        let insn1 = IRInsn::apply(
            "xor", vec![
                IROperand::reg("x1", IRValue::u5(8)), 
                IROperand::reg("x2", IRValue::u5(8)), 
                IROperand::reg("x3", IRValue::u5(9))
            ]
        );
        println!("{}", insn1);
    }

    #[test]
    fn ctx_info() {
        // println!("{}", IRContext::info());
        let ctx = IRContext::init();

        // Check pool info
        // println!("{}", IRContext::pool_info());

        let p0 = ctx.proc.borrow().clone();
        // Check process info
        println!("{}", p0.info());
        p0.new_thread();
        println!("{}", ctx.proc.borrow().info());

        p0.set_reg(3, IRValue::u32(23));
        println!("{}", ctx.proc.borrow().reg_info());
        println!("{}", p0.get_reg(2));


    }


}