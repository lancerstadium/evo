//! `evo::ir::op`: Opcodes and operands definition in the IR
//! 
//! 


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //



use crate::ir::val::IRValue;

use super::ty::IRType;


// ============================================================================== //
//                              op::IROperandKind
// ============================================================================== //

#[derive(Debug, Clone)]
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


    /// Get the size of the operand
    pub fn size(&self) -> usize {
        match self {
            IROperandKind::Imm(val) => val.size(),
            IROperandKind::Reg(val) => val.size(),
            IROperandKind::Mem(val) => val.size(),
            IROperandKind::Label(val) => val.size(),
        }
    }
}

// ============================================================================== //
//                               op::IROperand
// ============================================================================== //


/// `IROperand`: Operands in the IR
#[derive(Debug, Clone)]
pub struct IROperand {
    /// `kind`: Kind of the operand (Imm, Reg, Mem, Label)
    pub kind: IROperandKind,
}



// ============================================================================== //
//                                op::ArchInfo
// ============================================================================== //


/// `ArchInfo`: Config information of the architecture
pub trait ArchInfo {

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

    /// Regs' Hashmap: <RegName: String, RegValue: IROperand>
    const REG_MAP: &'static [(&'static str, IROperand)];

    /// Get Arch string
    fn to_string () -> String;


}




// ============================================================================== //
//                              op::IRArchInfo
// ============================================================================== //


/// `IRArchInfo`: Config of the `evo-ir` architecture
#[derive(Debug, Clone)]
pub struct IRArchInfo {


}

impl ArchInfo for IRArchInfo {

    // 1. Set Constants
    const NAME: &'static str = "evo";
    const BYTE_SIZE: usize = 1;
    const ADDR_SIZE: usize = 32;
    const WORD_SIZE: usize = 32;
    const FLOAT_SIZE: usize = 32;
    const REG_NUM: usize = 32;

    // 2. Set Register Map
    const REG_MAP: &'static [(&'static str, IROperand)] = &[
        // ("ax", IROperand::reg_init()),
    ];

    /// Get Arch string
    fn to_string () -> String {
        // Append '-' with the address size
        format!("{}-{}", Self::NAME, Self::ADDR_SIZE)
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
