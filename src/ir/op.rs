//! `evo::ir::op`: Opcodes and operands definition in the IR
//! 
//! 


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //



use crate::ir::val::IRValue;

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
