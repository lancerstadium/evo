
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use crate::arch::info::Arch;
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
    pub src_arch: &'static Arch,
    pub trg_arch: &'static Arch,

}


impl BasicBlock {


    pub fn init(src_insns: Vec<Instruction>, trg_arch: &'static Arch, flag: u16) -> BasicBlock {
        let src_arch = src_insns[0].arch;
        Self {
            flag,
            src_insns,
            trg_insns: vec![],
            src_arch,
            trg_arch,
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