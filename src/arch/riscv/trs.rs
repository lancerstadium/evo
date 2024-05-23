
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::vec;


use crate::log_error;
use crate::util::log::Span;
use crate::arch::info::Arch;
use crate::core::insn::{Instruction, RegFile};
use crate::core::trs::Translator;
use crate::arch::riscv::def::RISCV32_ARCH;
use crate::arch::evo::def::EVO_ARCH;



// ============================================================================== //
//                                 RISC-V 32
// ============================================================================== //

pub fn riscv32_trs_init(trg_arch: &'static Arch) -> Option<Rc<RefCell<Translator>>> {

    match *trg_arch {
        EVO_ARCH => {
            // bundle regs
            RegFile::reg_bundle(&RISCV32_ARCH, &EVO_ARCH, "x1", "t0");
            RegFile::reg_release(&RISCV32_ARCH, &EVO_ARCH, "x1");
            RegFile::reg_bundle(&RISCV32_ARCH, &EVO_ARCH, "x2", "t9");
            let trs = Translator::def(&RISCV32_ARCH, &EVO_ARCH);
            // define func
            trs.borrow_mut().def_func("add", 
                // ======== add rd, rs1, rs2 -> add_i32 rd, rs1, rs2 ======== //
                |insn| {
                    let mut trg_insns = Vec::new();
                    let insn1 = Instruction::insn_pool_nget("add_i32").borrow().clone().encode(vec![
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.rd() as usize).borrow().clone(),
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.rs1() as usize).borrow().clone(),
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.rs2() as usize).borrow().clone(),
                    ]);
                    trg_insns.push(insn1);
                    trg_insns
                }
            );
            trs.borrow_mut().def_func("sub", 
                // ======== add rd, rs1, rs2 -> add_i32 rd, rs1, rs2 ======== //
                |insn| {
                    let mut trg_insns = Vec::new();
                    let insn1 = Instruction::insn_pool_nget("sub_i32").borrow().clone().encode(vec![
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.rd() as usize).borrow().clone(),
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.rs1() as usize).borrow().clone(),
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.rs2() as usize).borrow().clone(),
                    ]);
                    trg_insns.push(insn1);
                    trg_insns
                }
            );
            Some(trs)
        },
        _ => {
            log_error!("Translator riscv32-{} not support", trg_arch);
            None
        }
    }
}



#[cfg(test)]
mod riscv_test {

    use super::*;
    use crate::core::itp::Interpreter;

    #[test]
    fn rv32_trs() {
        Interpreter::itp_pool_init(&RISCV32_ARCH);
        Interpreter::itp_pool_init(&EVO_ARCH);
        let trs = Translator::trs_pool_init(&RISCV32_ARCH, &EVO_ARCH).unwrap();
        println!("{}", Translator::func_pool_info());

        let src_insn1 = Instruction::from_string("add x0, x1, x2");
        let trg_insn1 = trs.borrow().translate(&src_insn1);
        println!("- src: \n{}\n- trg: \n{}", src_insn1, trg_insn1.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn2 = Instruction::from_string("sub x0, x1, x2");
        let trg_insn2 = trs.borrow().translate(&src_insn2);
        println!("- src: \n{}\n- trg: \n{}", src_insn2, trg_insn2.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));
    }


}