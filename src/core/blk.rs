
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::arch::info::Arch;
use crate::core::insn::Instruction;
use crate::core::trs::Translator;



// ============================================================================== //
//                                blk::BasicBlock
// ============================================================================== //


/// BasicBlock
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    
    pub flag: u16,
    pub src_insns: Vec<Instruction>,
    pub src_arch: &'static Arch,
    
    pub trg_insns: Option<Vec<Instruction>>,
    pub trg_arch: Option<&'static Arch>,
    pub trs_insn_idxs: Option<Vec<usize>>,
    pub trs: Option<Rc<RefCell<Translator>>>
}


impl BasicBlock {


    pub fn init(src_insns: Vec<Instruction>, trg_arch: Option<&'static Arch>, flag: u16) -> BasicBlock {
        let src_arch = src_insns[0].arch;
        if trg_arch.is_none() {
            Self {
                flag,
                src_insns,
                trg_insns: None,
                src_arch,
                trg_arch: None,
                trs_insn_idxs: None,
                trs: None
            }
        } else {
            Self {
                flag,
                src_insns,
                trg_insns: None,
                src_arch,
                trg_arch,
                trs_insn_idxs: None,
                trs: Translator::trs_pool_init(src_arch, trg_arch.unwrap())
            }
        }
    }

    pub fn is_translated(&self) -> bool {
        self.trg_insns.is_some()
    }

    pub fn translate(&mut self) {
        if self.trs.is_some() {
            let mut trg_insns = Vec::new();
            let mut trs_insn_idxs = Vec::new();
            let mut sum_insn_idx = 0;
            for src_insn in &self.src_insns {
                let mut trg_tmp_insns = self.trs.as_ref().unwrap().borrow_mut().translate(src_insn);
                trs_insn_idxs.push(sum_insn_idx);
                sum_insn_idx += trg_tmp_insns.len();
                trg_insns.append(&mut trg_tmp_insns);
            }
            self.trs_insn_idxs = Some(trs_insn_idxs);
            self.trg_insns = Some(trg_insns);
        }
    }

    pub fn trs_insn_info(&self, src_insn_idx: usize) -> String {
        if src_insn_idx >= self.src_insns.len() || !self.is_translated() {
            return "".to_string();
        } else {
            let mut info = String::new();
            let src_insn = &self.src_insns[src_insn_idx];
            let trs_insn_idx = self.trs_insn_idxs.as_ref().unwrap()[src_insn_idx];
            let trs_insn_num = if src_insn_idx + 1 < self.src_insns.len() {
                self.trs_insn_idxs.as_ref().unwrap()[src_insn_idx+1] - trs_insn_idx
            } else {
                self.trg_insns.as_ref().unwrap().len() - trs_insn_idx
            };
            let trs_insn = &self.trg_insns.as_ref().unwrap()[trs_insn_idx..trs_insn_idx+trs_insn_num];
            info.push_str(&format!(
                "[Src: {}, idx: {}]\n{}\n[Trg: {}, idx: {}]\n{}\n",
                self.src_arch, src_insn_idx, src_insn,
                self.trg_arch.unwrap(), trs_insn_idx, trs_insn.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n")
            ));
            info
        }
    }

    pub fn translate_info(&self, src_insn_idx: i32) -> String {
        if src_insn_idx < 0 || src_insn_idx >= self.src_insns.len() as i32 {
            let mut src_info = String::new();
            let mut trg_info = String::new();
            let mut info = String::new();
            src_info.push_str(&format!("[Src: {}, Nums: {}]: \n{}\n", self.src_arch, self.src_insns.len(), self.src_insns.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n")));
            trg_info.push_str(&format!("[Trg: {}, Nums: {}]: \n{}\n", self.trg_arch.unwrap(), self.trg_insns.as_ref().unwrap().len(), self.trg_insns.as_ref().unwrap().iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n")));
            info.push_str(&format!("{}\n{}\n", src_info, trg_info));
            info
        } else {
            self.trs_insn_info(src_insn_idx as usize)
        }
    }

}


#[cfg(test)]
mod blk_test {

    use super::*;
    use crate::core::insn::RegFile;
    use crate::core::itp::Interpreter;
    use crate::arch::riscv::def::RISCV32_ARCH;
    use crate::arch::evo::def::EVO_ARCH;

    #[test]
    fn blk_init() {
        Interpreter::itp_pool_init(&RISCV32_ARCH);
        Interpreter::itp_pool_init(&EVO_ARCH);

        // println!("{}", RegFile::reg_pool_info(&EVO_ARCH));

        let src_insn1 = Instruction::from_string("add x0, x1, x2");
        let src_insn2 = Instruction::from_string("sub x0, x1, x2");

        let src_insns = vec![src_insn1, src_insn2];
        let mut bb = BasicBlock::init(src_insns, Some(&EVO_ARCH), 0);
        println!("{}", RegFile::reg_pool_info(&EVO_ARCH));
        bb.translate();
        println!("{}", bb.translate_info(-1));
    }
}