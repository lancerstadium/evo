
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::vec;


use crate::core::cpu::CPUState;
use crate::log_error;
use crate::util::log::Span;
use crate::arch::info::Arch;
use crate::core::insn::{Instruction, RegFile, COND_EQ, COND_GE, COND_LT, COND_NE};
use crate::core::val::Value;
use crate::core::op::Operand;
use crate::core::trs::Translator;
use crate::arch::riscv::def::RISCV32_ARCH;
use crate::arch::evo::def::EVO_ARCH;
use crate::evo_gen;




macro_rules! trs_evo_r {
    ($trs:expr, $opcode:literal, $target_opcode:literal $(, $($params:expr),*)?) => {
        $trs.borrow_mut().def_func($opcode, 
            |_, insn| {
                let mut trg_insns = Vec::new();
                let insn1 = evo_gen!($target_opcode,
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(),
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[1].get_reg()).borrow().clone(),
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[2].get_reg()).borrow().clone()
                    $(, $($params),*)?
                );
                trg_insns.push(insn1);
                trg_insns
            }
        );
    };
}

macro_rules! trs_evo_i {
    ($trs:expr, $opcode:literal, $target_opcode:literal $(, $($params:expr),*)?) => {
        $trs.borrow_mut().def_func($opcode, 
            |_, insn| {
                let mut trg_insns = Vec::new();
                let insn1 = evo_gen!($target_opcode,
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(),
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[1].get_reg()).borrow().clone(),
                    insn.opr[2].clone()
                    $(, $($params),*)?
                );
                trg_insns.push(insn1);
                trg_insns
            }
        );
    };
}

macro_rules! trs_evo_b {
    ($trs:expr, $opcode:literal, $target_opcode:literal $(, $($params:expr),*)?) => {
        $trs.borrow_mut().def_func($opcode, 
            |_, insn| {
                let mut trg_insns = Vec::new();
                let insn1 = evo_gen!($target_opcode,
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(),
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[1].get_reg()).borrow().clone(),
                    insn.opr[2].clone()
                    $(, $($params),*)?
                );
                trg_insns.push(insn1);
                trg_insns
            }
        );
    };
}

macro_rules! u_evo_trs {
    ($trs:expr, $opcode:literal, $target_opcode:literal $(, $($params:expr),*)?) => {
        $trs.borrow_mut().def_func($opcode, 
            |_, insn| {
                let mut trg_insns = Vec::new();
                let insn1 = evo_gen!($target_opcode,
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(),
                    insn.opr[1].clone()
                    $(, $($params),*)?
                );
                trg_insns.push(insn1);
                trg_insns
            }
        );
    };
}

macro_rules! j_evo_trs {
    ($trs:expr, $opcode:literal, $target_opcode:literal $(, $($params:expr),*)?) => {
        $trs.borrow_mut().def_func($opcode, 
            |_, insn| {
                let mut trg_insns = Vec::new();
                let insn1 = evo_gen!($target_opcode,
                    RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(),
                    insn.opr[1].clone()
                    $(, $($params),*)?
                );
                trg_insns.push(insn1);
                trg_insns
            }
        );
    };
}




// ============================================================================== //
//                                 RISC-V 32
// ============================================================================== //

pub fn riscv32_trs_init(trg_arch: &'static Arch) -> Option<Rc<RefCell<Translator>>> {

    match *trg_arch {
        EVO_ARCH => {
            // bundle regs
            RegFile::reg_bund(&RISCV32_ARCH, &EVO_ARCH, "x1", "t0");
            RegFile::reg_free(&RISCV32_ARCH, &EVO_ARCH, "x1");
            RegFile::reg_bund(&RISCV32_ARCH, &EVO_ARCH, "x2", "t9");
            let trs = Translator::def(&RISCV32_ARCH, &EVO_ARCH);
            // Translate to evo: R Type
            trs_evo_r!(trs, "add", "add_i32");
            trs_evo_r!(trs, "sub", "sub_i32");
            trs_evo_r!(trs, "or" , "or_i32" );
            trs_evo_r!(trs, "and", "and_i32");
            trs_evo_r!(trs, "xor", "xor_i32");
            trs_evo_r!(trs, "sll", "shl_i32");
            trs_evo_r!(trs, "srl", "shr_i32");
            trs_evo_r!(trs, "sra", "sar_i32");
            trs_evo_r!(trs, "slt" , "cond_i32", Operand::imm(Value::u32(COND_LT as u32)));
            trs_evo_r!(trs, "sltu", "cond_u32", Operand::imm(Value::u32(COND_LT as u32)));

            // Translate to evo: I Type
            trs_evo_i!(trs, "addi", "add_i32");
            trs_evo_i!(trs, "ori" , "or_i32" );
            trs_evo_i!(trs, "andi", "and_i32");
            trs_evo_i!(trs, "xori", "xor_i32");
            trs_evo_i!(trs, "slti" , "cond_i32", Operand::imm(Value::u32(COND_LT as u32)));
            trs_evo_i!(trs, "sltiu", "cond_u32", Operand::imm(Value::u32(COND_LT as u32)));
            trs_evo_i!(trs, "lb"  , "ldb_i32");
            trs_evo_i!(trs, "lh"  , "ldh_i32");
            trs_evo_i!(trs, "lw"  , "ldw_i32");
            trs_evo_i!(trs, "lbu" , "ldb_u32");
            trs_evo_i!(trs, "lhu" , "ldh_u32");
            trs_evo_i!(trs, "sb"  , "stb_i32");
            trs_evo_i!(trs, "sh"  , "sth_i32");
            trs_evo_i!(trs, "sw"  , "stw_i32");

            trs.borrow_mut().def_func("ecall", 
                |_, insn| {
                    let mut trg_insns = Vec::new();
                    let insn1 = evo_gen!("syscall");
                    trg_insns.push(insn1);
                    trg_insns
                }
            );

            // Translate to evo: B Type
            trs_evo_b!(trs, "beq" , "cond_i32", Operand::imm(Value::u32(COND_EQ as u32)));
            trs_evo_b!(trs, "bne" , "cond_i32", Operand::imm(Value::u32(COND_NE as u32)));
            trs_evo_b!(trs, "blt" , "cond_i32", Operand::imm(Value::u32(COND_LT as u32)));
            trs_evo_b!(trs, "bge" , "cond_i32", Operand::imm(Value::u32(COND_GE as u32)));
            trs_evo_b!(trs, "bltu", "cond_u32", Operand::imm(Value::u32(COND_LT as u32)));
            trs_evo_b!(trs, "bgeu", "cond_u32", Operand::imm(Value::u32(COND_GE as u32)));

            // Translate to evo: U Type
            trs.borrow_mut().def_func("auipc", 
                |cpu, insn| {
                    let proc0 = cpu.proc.borrow().clone();
                    let mut trg_insns = Vec::new();
                    let insn1 = evo_gen!("mov_i32", 
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(), 
                        Operand::imm(Value::i32(insn.imm_u() as i32 + proc0.get_pc().get_i32(0)))
                    );
                    trg_insns.push(insn1);
                    trg_insns
                }
            );
            trs.borrow_mut().def_func("lui", 
                |cpu, insn| {
                    let proc0 = cpu.proc.borrow().clone();
                    let mut trg_insns = Vec::new();
                    let insn1 = evo_gen!("mov_i32", 
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(), 
                        Operand::imm(Value::i32((insn.imm_u() as i32) << 12))
                    );
                    trg_insns.push(insn1);
                    trg_insns
                }
            );

            // Translate to evo: J Type
            trs.borrow_mut().def_func("jal",
                |cpu, insn| {
                    let proc0 = cpu.proc.borrow().clone();
                    let mut trg_insns = Vec::new();
                    let insn1 = evo_gen!("mov_i32", 
                        RegFile::reg_alloc(&RISCV32_ARCH, &EVO_ARCH, insn.opr[0].get_reg()).borrow().clone(), 
                        Operand::imm(Value::i32(proc0.get_pc().get_i32(0) + proc0.get_pc_step() as i32))
                    );
                    let insn2 = evo_gen!("goto_tb", 
                        Operand::imm(Value::i32(0))
                    );
                    let insn3 = evo_gen!("mov_i32", 
                        RegFile::reg_poolr_nget(&EVO_ARCH, "pc").borrow().clone(),
                        Operand::imm(Value::i32(proc0.get_pc_next().get_i32(0) + insn.imm_u() as i32))
                    );
                    let insn4 = evo_gen!("exit_tb",
                        Operand::imm(Value::i32(0)),
                        Operand::imm(Value::i32(0))
                    );
                    trg_insns.push(insn1);
                    trg_insns.push(insn2);
                    trg_insns.push(insn3);
                    trg_insns.push(insn4);
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
    // 111110101011
    // 1110111110101011000000000000
    use super::*;
    use crate::core::cpu::CPUState;

    #[test]
    fn rv32_trs() {
        let mut cpu = CPUState::init(&RISCV32_ARCH, &EVO_ARCH, None, None, None);
        let trs = Translator::trs_pool_init(&RISCV32_ARCH, &EVO_ARCH).unwrap();
        println!("{}", Translator::func_pool_info());

        let src_insn1 = Instruction::from_string(&RISCV32_ARCH, "add x0, x1, x2");
        let trg_insn1 = trs.borrow().translate(&cpu, &src_insn1);
        println!("- src: \n{}\n- trg: \n{}", src_insn1, trg_insn1.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn2 = Instruction::from_string(&RISCV32_ARCH, "sub x0, x1, x2");
        let trg_insn2 = trs.borrow().translate(&cpu, &src_insn2);
        println!("- src: \n{}\n- trg: \n{}", src_insn2, trg_insn2.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn3 = Instruction::from_string(&RISCV32_ARCH, "addi x0, x1, 13");
        let trg_insn3 = trs.borrow().translate(&cpu, &src_insn3);
        println!("- src: \n{}\n- trg: \n{}", src_insn3, trg_insn3.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn4 = Instruction::from_string(&RISCV32_ARCH, "slt x0, x1, x2");
        let trg_insn4 = trs.borrow().translate(&cpu, &src_insn4);
        println!("- src: \n{}\n- trg: \n{}", src_insn4, trg_insn4.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn5 = Instruction::from_string(&RISCV32_ARCH, "slti x0, x1, 13");
        let trg_insn5 = trs.borrow().translate(&cpu, &src_insn5);
        println!("- src: \n{}\n- trg: \n{}", src_insn5, trg_insn5.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn6 = Instruction::from_string(&RISCV32_ARCH, "lh x0, x1, 143");
        let trg_insn6 = trs.borrow().translate(&cpu, &src_insn6);
        println!("- src: \n{}\n- trg: \n{}", src_insn6, trg_insn6.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

        let src_insn7 = Instruction::from_string(&RISCV32_ARCH, "lui x1, 0xab ef");
        let trg_insn7 = trs.borrow().translate(&cpu, &src_insn7);
        println!("- src: \n{}\n- trg: \n{}", src_insn7, trg_insn7.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));   

        let src_insn8 = Instruction::from_string(&RISCV32_ARCH, "jal x2, 0x21");
        let trg_insn8 = trs.borrow().translate(&cpu, &src_insn8);
        println!("- src: \n{}\n- trg: \n{}", src_insn8, trg_insn8.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("\n"));

    }


}