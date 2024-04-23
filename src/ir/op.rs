//! `evo::ir::op`: Opcodes and operands definition in the IR
//! 
//! 


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::fmt::{self};
use std::cmp;
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

    /// |  name*  |  val   |
    Reg(&'static str, IRValue),

    /// Mem = [base + index * scale + disp]
    /// |  base  |  idx  |  scala  | disp  |
    Mem(IRValue, IRValue, IRValue, IRValue),

    /// |  name*  |  addr  |
    Label(&'static str, IRValue),
}

impl IROperandKind {

    /// Get the size of the operand
    pub fn size(&self) -> usize {
        match self {
            IROperandKind::Imm(val) => val.size(),
            // Get Reg value's size
            IROperandKind::Reg(_, val) => val.size(),
            IROperandKind::Mem(val, _, _, _) => val.size(),
            // Get Label ptr's size
            IROperandKind::Label(_, val) => val.size(),
        }
    }

    /// Get hex of the operand
    pub fn hex(&self) -> String {
        match self {
            IROperandKind::Imm(val) => val.hex(0, -1),
            // Get Reg value's hex
            IROperandKind::Reg(_, val) => val.hex(0, -1),
            IROperandKind::Mem(_, _, _, _) => self.val().hex(0, -1),
            // Get Label ptr's hex
            IROperandKind::Label(_, val) => val.hex(0, -1),
        }
    }

    /// Get name of the operand
    pub fn name(&self) -> &'static str {
        match self {
            IROperandKind::Imm(_) => "<Imm>",
            IROperandKind::Reg(name, _) => name,
            IROperandKind::Mem(_, _, _, _) => "<Mem>",
            IROperandKind::Label(name, _) => name,
        }
    }

    /// Get String of the operand
    pub fn to_string(&self) -> String {
        match self {
            IROperandKind::Imm(val) => val.to_string(),
            IROperandKind::Reg(name, val) => format!("{}: {}", name, val.to_string()),
            IROperandKind::Mem(base, idx, scale, disp) => format!("[{} + {} * {} + {}]", base, idx, scale, disp),
            IROperandKind::Label(name, val) => format!("{}: {}", name, val.to_string()),
        }
    }

    /// Get value of the operand
    pub fn val(&self) -> IRValue {
        match self {
            IROperandKind::Imm(val) => val.clone(),
            IROperandKind::Reg(_, val) => val.clone(),
            IROperandKind::Mem(base, idx, scale, disp) => IRValue::u32(base.get_u32(0) + idx.get_u32(0) * scale.get_u32(0) + disp.get_u32(0)),
            IROperandKind::Label(_, val) => val.clone(),
        }
    }

}


impl fmt::Display for IROperandKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ============================================================================== //
//                               op::IROperand
// ============================================================================== //


/// `IROperand`: Operands in the IR
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IROperand {
    /// `kind`: Kind of the operand (Imm, Reg, Mem, Label)
    pub kind: IROperandKind,
}

impl IROperand {

    /// New IROperand Reg
    pub fn reg(name: &'static str, val: IRValue) -> Self {
        Self {
            kind: IROperandKind::Reg(name, val),
        }
    }


    // ================== IROperand.get ==================== //

    /// Get Operand value
    pub fn val(&self) -> IRValue {
        self.kind.val()
    }

    /// Get Operand name
    pub fn name(&self) -> &'static str {
        self.kind.name()
    }

    // ================== IROperand.set ==================== //

    /// Set Operand value
    pub fn set_reg(&mut self, val: IRValue) {
        self.kind = IROperandKind::Reg(self.kind.name(), val);
    }

}

impl fmt::Display for IROperand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.kind.to_string())
    }
}



// ============================================================================== //
//                              op::IROpcodeKind
// ============================================================================== //

/// `IROpcodeKind`: Kind of IR opcodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROpcodeKind {
    /// Special opcode: [opcode]
    Special(),

    /// Unary operand opcode: [opcode] [operand]
    Unary(IROperand),

    /// Binary operand opcode: [opcode] [operand1], [operand2]
    Binary(IROperand, IROperand),

    /// Ternary operand opcode: [opcode] [operand1], [operand2], [operand3]
    Ternary(IROperand, IROperand, IROperand),

    /// Quaternary operand opcode: [opcode] [operand1], [operand2], [operand3], [operand4]
    Quaternary(IROperand, IROperand, IROperand, IROperand),

}

impl fmt::Display for IROpcodeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ============================================================================== //
//                               op::IROpcode
// ============================================================================== //


/// `IROpcode`: IR opcodes
#[derive(Debug, Clone)]
pub struct IROpcode {
    /// `kind`: Kind of the opcode
    pub kind: IROpcodeKind,
}

impl IROpcode {

    /// New IROpcode Special
    pub fn special() -> Self {
        Self {
            kind: IROpcodeKind::Special(),
        }
    }

    /// New IROpcode Unary
    pub fn unary(operand: IROperand) -> Self {
        Self {
            kind: IROpcodeKind::Unary(operand),
        }
    }

    /// New IROpcode Binary
    pub fn binary(operand1: IROperand, operand2: IROperand) -> Self {
        Self {
            kind: IROpcodeKind::Binary(operand1, operand2),
        }
    }

    /// New IROpcode Ternary
    pub fn ternary(operand1: IROperand, operand2: IROperand, operand3: IROperand) -> Self {
        Self {
            kind: IROpcodeKind::Ternary(operand1, operand2, operand3),
        }
    }

    /// New IROpcode Quaternary
    pub fn quaternary(operand1: IROperand, operand2: IROperand, operand3: IROperand, operand4: IROperand) -> Self {
        Self {
            kind: IROpcodeKind::Quaternary(operand1, operand2, operand3, operand4),
        }
    }


}

impl fmt::Display for IROpcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.kind.to_string())
    }
}

impl cmp::PartialEq for IROpcode {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
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

    /// Register Map Init
    fn reg_init(&mut self);

    /// Reg Info: `RegName: RegValue` String
    fn reg_info(&self) -> String;

    /// Get Register
    fn get_reg(&self, name: &'static str) -> IRValue;

    /// Set Register
    fn set_reg(&mut self, name: &'static str, value: IRValue);

}




// ============================================================================== //
//                              op::IRArch
// ============================================================================== //


/// `IRArch`: Config of the `evo-ir` architecture
#[derive(Debug, Clone, PartialEq)]
pub struct IRArch {
    /// `reg_map`: Register Map
    reg_map: Rc<RefCell<Vec<IROperand>>>,
    /// `opcode_map`: Opcode Map
    opcode_map: Rc<RefCell<Vec<(&'static str, IROpcode)>>>,

}

impl IRArch {

    /// Create new `IRArch`
    pub fn new() -> Self {
        let mut arch = Self {
            reg_map: Rc::new(RefCell::new(Vec::new())),
            opcode_map: Rc::new(RefCell::new(Vec::new())),
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
            IROperand::reg("eax", IRValue::u32(0)),
            IROperand::reg("ebx", IRValue::u32(0)),
            IROperand::reg("ecx", IRValue::u32(0)),
            IROperand::reg("edx", IRValue::u32(0)),
            IROperand::reg("esi", IRValue::u32(0)),
            IROperand::reg("edi", IRValue::u32(0)),
            IROperand::reg("esp", IRValue::u32(0)),
            IROperand::reg("ebp", IRValue::u32(0)),
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
            result.push_str(&format!(" - {}\n", reg));
        }
        result
    }

    /// 6. Get Register
    fn get_reg(&self, name: &'static str) -> IRValue {
        // Get value according to name
        self.reg_map.borrow().iter().find(|reg| reg.name() == name).unwrap_or(&IROperand::reg(name, IRValue::u32(0))).val()
    }
    /// 7. Set Register
    fn set_reg(&mut self, name: &'static str, value: IRValue) {
        // Set value according to name and value
        self.reg_map.borrow_mut().iter_mut().find(|reg| reg.name() == name).unwrap_or(&mut IROperand::reg(name, IRValue::u32(0))).set_reg(value);
    }

}







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
        
        arch.set_reg("ebx", IRValue::u32(9));
        arch.set_reg("eax", IRValue::u32(8));
        assert_eq!(arch.get_reg("ebx"), IRValue::u32(9));
        println!("{}", arch.reg_info());

        let arch2 = IRArch::new();
        // Compare Registers
        assert_ne!(arch, arch2);
    }
}