
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use crate::core::insn::Instruction;
use crate::core::op::Operand;



// ============================================================================== //
//                                insn::BasicBlock
// ============================================================================== //


/// BasicBlock
pub struct BasicBlock {
    
    pub flag: u16,
    pub src_insns: Vec<Instruction>,
    pub ir_insns: Vec<Instruction>,
    pub trg_insns: Vec<Instruction>,
    pub liveness_regs: Vec<Operand>,

    /// If the block is lifted to EVO ir arch: src -> ir
    pub is_lifted: bool,
    /// If the block is lowered to target arch: ir -> trg
    pub is_lowered: bool,
}


impl BasicBlock {


    pub fn init(src_insns: Vec<Instruction>, flag: u16) -> BasicBlock {
        Self {
            flag,
            src_insns,
            ir_insns: vec![],
            trg_insns: vec![],
            liveness_regs: vec![],
            is_lifted: false,
            is_lowered: false
        }
    }

}


#[cfg(test)]
mod blk_test {

    use super::*;

    #[test]
    fn init() {
        
    }
}