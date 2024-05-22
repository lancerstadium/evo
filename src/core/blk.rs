
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use crate::core::insn::Instruction;
use crate::core::op::Operand;



// ============================================================================== //
//                                blk::BasicBlock
// ============================================================================== //


/// BasicBlock
pub struct BasicBlock {
    
    pub flag: u16,
    pub src_insns: Vec<Instruction>,
    pub trg_insns: Vec<Instruction>,

}


impl BasicBlock {


    pub fn init(src_insns: Vec<Instruction>, flag: u16) -> BasicBlock {
        Self {
            flag,
            src_insns,
            trg_insns: vec![],
        }
    }



}


#[cfg(test)]
mod blk_test {

    use super::*;

    #[test]
    fn blk_init() {
        
    }
}