//! `evo::ir::op`: Opcodes and operands definition in the IR
//! 
//! 


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::fmt::{self};
use std::rc::Rc;
use std::cell::RefCell;

use crate::ir::val::IRValue;
use crate::log_warning;
use crate::util::log::Span;



// ============================================================================== //
//                              op::IROperandKind
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROperandKind {
    /// |  val  |
    Imm(IRValue),

    /// |  val  |
    Reg(IRValue),

    /// Mem = [base + index * scale + disp]
    /// |  base  |  idx  |  scala  | disp  |
    // Mem(IRValue, IRValue, IRValue, IRValue),
    Mem(IRValue),

    /// |  addr  |
    Label(IRValue),
}

impl IROperandKind {

    /// New IROperand
    pub fn new_reg(val: IRValue) -> Self {
        Self::Reg(val)
    }

    /// Get the size of the operand
    pub fn size(&self) -> usize {
        match self {
            IROperandKind::Imm(val) => val.size(),
            IROperandKind::Reg(val) => val.size(),
            IROperandKind::Mem(val) => val.size(),
            IROperandKind::Label(val) => val.size(),
        }
    }

    /// Get hex of the operand
    pub fn hex(&self) -> String {
        match self {
            IROperandKind::Imm(val) => val.hex(),
            IROperandKind::Reg(val) => val.hex(),
            IROperandKind::Mem(val) => val.hex(),
            IROperandKind::Label(val) => val.hex(),
        }
    }

    /// Get String of the operand
    pub fn to_string(&self) -> String {
        match self {
            IROperandKind::Imm(val) => val.to_string(),
            IROperandKind::Reg(val) => val.to_string(),
            IROperandKind::Mem(val) => val.to_string(),
            IROperandKind::Label(val) => val.to_string(),
        }
    }

}

// ============================================================================== //
//                               op::IROperand
// ============================================================================== //


/// `IROperand`: Operands in the IR
#[derive(Debug, Clone, PartialEq)]
pub struct IROperand {
    /// `kind`: Kind of the operand (Imm, Reg, Mem, Label)
    pub kind: IROperandKind,
}

impl IROperand {

    /// New IROperand Reg
    pub fn new_reg(val: IRValue) -> Self {
        Self {
            kind: IROperandKind::new_reg(val),
        }
    }

}

impl fmt::Display for IROperand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.kind.to_string())
    }
}


// ============================================================================== //
//                                op::ArchInfo
// ============================================================================== //


/// `ArchInfo`: Config information of the architecture
pub trait ArchInfo {


    // ===================== Struct ====================== //

    /// Arch name: like "evo"
    const NAME: &'static str;

    /// Number of bytes in a byte: *1*, 2, 4
    const BYTE_SIZE: usize;

    /// Number of bytes in a addr(ptr/reg.size: 0x00 ~ 2^ADDR_SIZE): 8, 16, *32*, 64
    const ADDR_SIZE: usize;

    /// Number of bytes in a word(interger): 8, 16, *32*, 64
    const WORD_SIZE: usize;

    /// Number of bytes in a (float): *32*, 64
    const FLOAT_SIZE: usize;

    /// Number of Registers: 8, 16, *32*, 64
    const REG_NUM: usize;

    /// Get Arch string
    fn to_string () -> String;

    /// Get Info String
    fn info() -> String;

    // ===================== Object ====================== //

    /// Get Name
    fn name() -> &'static str;

    /// Register Map Init: Rc<RefCell<Vec<(&'static str, IROperand)>>>
    fn reg_init(&mut self);

    /// Reg Info: `RegName: RegValue` String
    fn reg_info(&self) -> String;

    /// Get Register
    fn get_reg(&self, name: &'static str) -> IROperand;

    /// Set Register
    fn set_reg(&mut self, name: &'static str, value: IROperand);

}




// ============================================================================== //
//                              op::IRArch
// ============================================================================== //


/// `IRArch`: Config of the `evo-ir` architecture
#[derive(Debug, Clone, PartialEq)]
pub struct IRArch {

    reg_map: Rc<RefCell<Vec<(&'static str, IROperand)>>>,

}

impl IRArch {

    /// Create new `IRArch`
    pub fn new() -> Self {
        let mut arch = Self {
            reg_map: Rc::new(RefCell::new(Vec::new())),
        };
        arch.reg_init();
        arch
    }
    
}


impl Default for IRArch {
    /// Set default function for `IRArch`.
    fn default() -> Self {
        Self::new()
    }
}


impl ArchInfo for IRArch {

    // 1. Set Constants
    const NAME: &'static str = "evo";
    const BYTE_SIZE: usize = 1;
    const ADDR_SIZE: usize = 32;
    const WORD_SIZE: usize = 32;
    const FLOAT_SIZE: usize = 32;
    const REG_NUM: usize = 32;

    /// 2. Get Arch string
    fn to_string () -> String {
        // Append '-' with the address size
        format!("{}-{}", Self::NAME, Self::ADDR_SIZE)
    }

    /// 3. Get ArchInfo string
    fn info() -> String {
        format!("[{}]:\n - byte: {}\n - addr: {}\n - word: {}\n - float: {}\n - reg: {}", Self::to_string(), Self::BYTE_SIZE, Self::ADDR_SIZE, Self::WORD_SIZE, Self::FLOAT_SIZE, Self::REG_NUM)
    }


    /// 3. Get Name
    fn name() -> &'static str {
        Self::NAME
    }

    /// 4. Register Map Init
    fn reg_init(&mut self) {
        self.reg_map = Rc::new(vec![
            ("eax", IROperand::new_reg(IRValue::from_u32(0))),
            ("ebx", IROperand::new_reg(IRValue::from_u32(1))),
            ("ecx", IROperand::new_reg(IRValue::from_u32(2))),
            ("edx", IROperand::new_reg(IRValue::from_u32(3))),
            ("esi", IROperand::new_reg(IRValue::from_u32(4))),
            ("edi", IROperand::new_reg(IRValue::from_u32(5))),
            ("ebp", IROperand::new_reg(IRValue::from_u32(6))),
            ("esp", IROperand::new_reg(IRValue::from_u32(7))),
        ].into());
        if Self::ADDR_SIZE != self.reg_map.borrow().len() {
            log_warning!("Register map not match with address size: {} != {}", self.reg_map.borrow().len() , Self::ADDR_SIZE);
        }
    }

    /// 5. Reg Info
    fn reg_info(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!("Regs Info: \n"));
        for reg in self.reg_map.borrow().iter() {
            result.push_str(&format!(" - {}: {}\n", reg.0, reg.1));
        }
        result
    }

    /// 6. Get Register
    fn get_reg(&self, name: &'static str) -> IROperand {
        self.reg_map.borrow().iter().find(|reg| reg.0 == name).unwrap().1.clone()
    }

    /// 7. Set Register
    fn set_reg(&mut self, name: &'static str, value: IROperand) {
        self.reg_map.borrow_mut().iter_mut().find(|reg| reg.0 == name).unwrap().1 = value;
    }

}






// ============================================================================== //
//                              op::IROpcodeKind
// ============================================================================== //

/// `IROpcodeKind`: Kind of IR opcodes
#[derive(Debug, Clone)]
pub enum IROpcodeKind {

}

// ============================================================================== //
//                               op::IROpcode
// ============================================================================== //







// ============================================================================== //
//                                Unit Tests
// ============================================================================== //

#[cfg(test)]
mod op_test {

    use super::*;

    #[test]
    fn arch_print() {
        println!("{}", IRArch::info());
        let mut arch = IRArch::new();
        println!("{}", arch.reg_info());
        arch.set_reg("eax", IROperand::new_reg(IRValue::from_u32(72)));
        assert_eq!(arch.get_reg("ebx"), IROperand::new_reg(IRValue::from_u32(1)));

        let arch2 = IRArch::new();
        // Compare Registers
        assert_ne!(arch, arch2);
    }
}