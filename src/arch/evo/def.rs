

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::{log_error, log_info};
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT32, BIT64, LITTLE_ENDIAN};
use crate::core::val::Value;
use crate::core::op::{OpcodeKind, Operand, OperandKind, OPR_IMM, OPR_LAB, OPR_OFF, OPR_REG, OPR_UND};
use crate::core::insn::{Instruction, RegFile, COND_AL, COND_EQ, COND_GE, COND_GT, COND_LE, COND_LT, COND_NE, COND_NO, INSN_SIG, INSN_USD};
use crate::core::itp::Interpreter;



// ============================================================================== //
//                             evo::def::arch
// ============================================================================== //

pub const EVO_ARCH: Arch = Arch::new(ArchKind::EVO, BIT64 | LITTLE_ENDIAN, 256);




// ============================================================================== //
//                          evo::def::interpreter
// ============================================================================== //

/// Insn temp and Reg and Interpreter Pool Init
pub fn evo_itp_init() -> Option<Rc<RefCell<Interpreter>>> {

    // 1. Init regs pool
    RegFile::def(&EVO_ARCH, "t0", Value::bit(8, 0), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t1", Value::bit(8, 1), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t2", Value::bit(8, 2), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t3", Value::bit(8, 3), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t4", Value::bit(8, 4), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t5", Value::bit(8, 5), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t6", Value::bit(8, 6), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t7", Value::bit(8, 7), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t8", Value::bit(8, 8), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t9", Value::bit(8, 9), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t10", Value::bit(8, 10), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t11", Value::bit(8, 11), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t12", Value::bit(8, 12), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t13", Value::bit(8, 13), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t14", Value::bit(8, 14), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t15", Value::bit(8, 15), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t16", Value::bit(8, 16), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t17", Value::bit(8, 17), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t18", Value::bit(8, 18), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t19", Value::bit(8, 19), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t20", Value::bit(8, 20), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t21", Value::bit(8, 21), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t22", Value::bit(8, 22), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t23", Value::bit(8, 23), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t24", Value::bit(8, 24), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t25", Value::bit(8, 25), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t26", Value::bit(8, 26), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t27", Value::bit(8, 27), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t28", Value::bit(8, 28), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t29", Value::bit(8, 29), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t30", Value::bit(8, 30), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t31", Value::bit(8, 31), BIT64 | LITTLE_ENDIAN);


    RegFile::def(&EVO_ARCH, "pc", Value::bit(8, 64), BIT64 | LITTLE_ENDIAN);

    // 2. Init insns & insns interpreter
    let itp = Interpreter::def(&EVO_ARCH);

    // ============================================================================== //
    //                              Arithmetic Instructions
    // ============================================================================== //

    itp.borrow_mut().def_insn("nop" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![], "E", "0x00",
        |_, _| {
            // ======== No Operation ======== //
        }
    );

    itp.borrow_mut().def_insn("add_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG,  vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x01", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Add rs1 and rs2
            let res = rs1.wrapping_add(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("add_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG,  vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x01", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Add rs1 and rs2
            let res = rs1.wrapping_add(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("sub_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x02", 
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Sub rs1 and rs2
            let res = rs1.wrapping_sub(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("sub_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x02", 
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Sub rs1 and rs2
            let res = rs1.wrapping_sub(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("neg_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x03", 
        |cpu, insn| {
            // ======== rd = -rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Neg rs1
            let res = rs1.wrapping_neg();
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("neg_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x03", 
        |cpu, insn| {
            // ======== rd = -rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Neg rs1
            let res = rs1.wrapping_neg();
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("mul_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG,  vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x04", 
        |cpu, insn| {
            // ======== rd = rs1 * rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Mul rs1 and rs2
            let res = rs1.wrapping_mul(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("mul_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x04", 
        |cpu, insn| {
            // ======== rd = rs1 * rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Mul rs1 and rs2
            let res = rs1.wrapping_mul(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("div_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x05", 
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("div_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x05", 
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("div_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x05", 
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Get rs2(u32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u32(0)
            } else {
                0
            };
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("div_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x05", 
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u64(0)
            } else {
                0
            };
            // 2. Get rs2(u64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u64(0)
            } else {
                0
            };
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("rem_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x06", 
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("rem_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x06", 
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("rem_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x06", 
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Get rs2(u32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u32(0)
            } else {
                0
            };
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("rem_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x06", 
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u64(0)
            } else {
                0
            };
            // 2. Get rs2(u64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u64(0)
            } else {
                0
            };
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );

    // ============================================================================== //
    //                              Bitwise Instructions
    // ============================================================================== //

    itp.borrow_mut().def_insn("and_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x07", 
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. And rs1 and rs2
            let res = rs1 & rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("and_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x07", 
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. And rs1 and rs2
            let res = rs1 & rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("or_i32"  , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x08", 
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Or rs1 and rs2
            let res = rs1 | rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("or_i64"  , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x08", 
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Or rs1 and rs2
            let res = rs1 | rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("xor_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x09", 
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Xor rs1 and rs2
            let res = rs1 ^ rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("xor_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x09", 
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Xor rs1 and rs2
            let res = rs1 ^ rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("not_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x0a", 
        |cpu, insn| {
            // ======== rd = ~rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Not rs1
            let res = !rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("not_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x0a", 
        |cpu, insn| {
            // ======== rd = ~rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Not rs1
            let res = !rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("andc_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0b", 
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Addc rs1 and rs2
            let res = rs1 & !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("andc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0b", 
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Addc rs1 and rs2
            let res = rs1 & !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("eqv_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0c", 
        |cpu, insn| {
            // ======== rd = rs1 ^ ~rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Eqv rs1 and rs2
            let res = rs1 ^ !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("eqv_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0c", 
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Eqv rs1 and rs2
            let res = rs1 ^ !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("nand_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0d", 
        |cpu, insn| {
            // ======== rd = ~(rs1 & rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Nand rs1 and rs2
            let res = !(rs1 & rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("nand_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0d", 
        |cpu, insn| {
            // ======== rd = ~(rs1 & rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Nand rs1 and rs2
            let res = !(rs1 & rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("nor_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0e", 
        |cpu, insn| {
            // ======== rd = ~(rs1 | rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Nor rs1 and rs2
            let res = !(rs1 | rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("nor_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0e", 
        |cpu, insn| {
            // ======== rd = ~(rs1 | rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Nor rs1 and rs2
            let res = !(rs1 | rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("orc_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0f", 
        |cpu, insn| {
            // ======== rd = rs1 | ~rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Orc rs1 and rs2
            let res = rs1 | !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("orc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x0f", 
        |cpu, insn| {
            // ======== rd = rs1 | ~rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Orc rs1 and rs2
            let res = rs1 | !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("clz_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x10", 
        |cpu, insn| {
            // ======== rd = rs1 ? clz(rs1) : rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Clz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.leading_zeros() as i32 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("clz_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x10", 
        |cpu, insn| {
            // ======== rd = rs1 ? clz(rs1) : rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Clz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.leading_zeros() as i64 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("ctz_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x11", 
        |cpu, insn| {
            // ======== rd = rs1 ? ctz(rs1) : rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Ctz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.trailing_zeros() as i32 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("ctz_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x11", 
        |cpu, insn| {
            // ======== rd = rs1 ? ctz(rs1) : rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Ctz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.trailing_zeros() as i64 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("shl_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x12", 
        |cpu, insn| {
            // ======== rd = rs1 << rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Shl rs1 and rs2
            let res = rs1 << rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("shl_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x12", 
        |cpu, insn| {
            // ======== rd = rs1 << rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Shl rs1 and rs2
            let res = rs1 << rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("shr_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x13", 
        |cpu, insn| {
            // ======== rd = rs1(u32) >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Shr rs1 and rs2
            let res = rs1 >> rs2;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("shr_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x13", 
        |cpu, insn| {
            // ======== rd = rs1(u64) >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Shr rs1 and rs2
            let res = rs1 >> rs2;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("sar_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x14", 
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Sar rs1 and rs2
            let res = rs1 >> rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("sar_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x14", 
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Sar rs1 and rs2
            let res = rs1 >> rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("rol_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x15", 
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Rotl rs1 and rs2
            let res = rs1.rotate_left(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("rol_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG,vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x15", 
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Rotl rs1 and rs2
            let res = rs1.rotate_left(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("ror_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x16", 
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Rotr rs1 and rs2
            let res = rs1.rotate_right(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("ror_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x16", 
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Rotr rs1 and rs2
            let res = rs1.rotate_right(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );

    // ============================================================================== //
    //                           Typetransfer Instructions
    // ============================================================================== //

    itp.borrow_mut().def_insn("mov_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x17", 
        |cpu, insn| {
            // ======== rd = rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Mov rs1
            let res = rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("mov_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x17", 
        |cpu, insn| {
            // ======== rd = rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Mov rs1
            let res = rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extb_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x18",
        |cpu, insn| {
            // ======== rd = rs1(i8 -> i32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i8)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i8(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i8(0)
            } else {
                0
            };
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("extb_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x18",
        |cpu, insn| {
            // ======== rd = rs1(i8 -> i64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i8)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i8(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i8(0)
            } else {
                0
            };
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extb_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x18",
        |cpu, insn| {
            // ======== rd = rs1(u8 -> u32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u8)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u8(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u8(0)
            } else {
                0
            };
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("extb_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x18",
        |cpu, insn| {
            // ======== rd = rs1(u8 -> u64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u8)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u8(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u8(0)
            } else {
                0
            };
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("exth_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x19",
        |cpu, insn| {
            // ======== rd = rs1(i16 -> i32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i16(0)
            } else {
                0
            };
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("exth_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x19",
        |cpu, insn| {
            // ======== rd = rs1(i16 -> i64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i16(0)
            } else {
                0
            };
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("exth_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x19",
        |cpu, insn| {
            // ======== rd = rs1(u16 -> u32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(0)
            } else {
                0
            };
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("exth_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x19",
        |cpu, insn| {
            // ======== rd = rs1(u16 -> u64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(0)
            } else {
                0
            };
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("extw_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1a",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> i32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("extw_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1a",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> i64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extw_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1a",
        |cpu, insn| {
            // ======== rd = rs1(u32 -> u32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("extw_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1a",
        |cpu, insn| {
            // ======== rd = rs1(u32 -> u64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("etr_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_IMM, OPR_IMM], "E", "0x1b",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get imm1(i32)
            let imm1 = insn.opr[2].val().get_i32(0);
            // 3. Get imm2(i32)
            let imm2 = insn.opr[3].val().get_i32(0);
            // 4. Extract rs1(pos, len) to i32
            let res = (rs1 >> imm1) & ((1 << imm2) - 1);
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("etr_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_IMM, OPR_IMM], "E", "0x1b",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get imm1(i64)
            let imm1 = insn.opr[2].val().get_i64(0);
            // 3. Get imm2(i64)
            let imm2 = insn.opr[3].val().get_i64(0);
            // 4. Extract rs1(pos, len) to i32
            let res = (rs1 >> imm1) & ((1 << imm2) - 1);
            // 5. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("etr_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_IMM, OPR_IMM], "E", "0x1b",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Get imm1(i32)
            let imm1 = insn.opr[2].val().get_i32(0);
            // 3. Get imm2(i32)
            let imm2 = insn.opr[3].val().get_i32(0);
            // 4. Extract rs1(pos, len) to i32
            let res = (rs1 >> imm1) & ((1 << imm2) - 1);
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("etr_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_IMM, OPR_IMM], "E", "0x1b",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u64(0)
            } else {
                0
            };
            // 2. Get imm1(i64)
            let imm1 = insn.opr[2].val().get_i64(0);
            // 3. Get imm2(i64)
            let imm2 = insn.opr[3].val().get_i64(0);
            // 4. Extract rs1(pos, len) to i32
            let res = (rs1 >> imm1) & ((1 << imm2) - 1);
            // 5. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );

    itp.borrow_mut().def_insn("etrl_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1c",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> low half) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(0)
            } else {
                0
            };
            // 2. Set rd(i64[15:0])
            let mut rd = proc0.get_reg(insn.opr[0].val().get_byte(0) as usize);
            rd.set_uhalf(0, 0, 16, rs1);
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("etrl_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1c",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> low word) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Set rd(i64[31:0])
            let mut rd = proc0.get_reg(insn.opr[0].val().get_byte(0) as usize);
            rd.set_uword(0, 0, 32, rs1);
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("etrh_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1d",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> high half) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(2)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(2)
            } else {
                0
            };
            // 2. Set rd(i64[15:0])
            let mut rd = proc0.get_reg(insn.opr[0].val().get_byte(0) as usize);
            rd.set_uhalf(0, 0, 16, rs1);
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("etrh_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1d",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> high word) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(4)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(4)
            } else {
                0
            };
            // 2. Set rd(i64[31:0])
            let mut rd = proc0.get_reg(insn.opr[0].val().get_byte(0) as usize);
            rd.set_uword(0, 0, 32, rs1);
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("truc_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1e",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> low half + fill 0) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(0)
            } else {
                0
            };
            // 2. set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(rs1 as u64));
        }
    );
    itp.borrow_mut().def_insn("truc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x1e",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> low word + fill 0) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(rs1 as u64));
        }
    );
    itp.borrow_mut().def_insn("conc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x1f",
        |cpu, insn| {
            // ======== rd = (rs2 << 32) | rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            } as u64;
            // 2. Get rs2(u32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u32(0)
            } else {
                0
            } as u64;
            // 3. Concat rs1 low 32-bit word and rs2 high 32-bit word
            let res = (rs2 << 32) | rs1;
            // 3. set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("conn_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x20",
        |cpu, insn| {
            // ======== rd = rs1(low word) | rs2(high word) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            } as u64;
            // 2. Get rs2(u32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u32(4)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u32(4)
            } else {
                0
            } as u64;
            // 3. Concat rs1 low 32-bit word and rs2 high 32-bit word
            let res = (rs2 << 32) | rs1;
            // 3. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("swph_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x21",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(0)
            } else {
                0
            };
            // 2. Reverse byte
            let res = u16::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swph_i64" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x21",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u16(0)
            } else {
                0
            };
            // 2. Reverse byte
            let res = u16::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swpw_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x22",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Reverse byte
            let res = u32::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swpw_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x22",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Reverse byte
            let res = u32::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swpd_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM], "E", "0x23",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u64(0)
            } else {
                0
            };
            // 2. Reverse byte
            let res = u64::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("depo_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM], "E", "0x24",
        |cpu, insn| {
            // ======== rd, rs1, rs2, ofs, len ======== //
            log_info!("TODO: depo_i32 rd, rs1, rs2, ofs, len");
        }
    );
    itp.borrow_mut().def_insn("depo_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM], "E", "0x24",
        |cpu, insn| {
            // ======== rd, rs1, rs2, ofs, len ======== //
            log_info!("TODO: depo_i64 rd, rs1, rs2, ofs, len");
        }
    );


    // ============================================================================== //
    //                             MemIO Instructions
    // ============================================================================== //

    itp.borrow_mut().def_insn("ldb_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x25",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_byte(0) as i32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldb_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x25",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_byte(0) as i64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldb_u32", BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x25",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_byte(0) as u32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldb_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x25",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_byte(0) as u64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val));
        }
    );
    itp.borrow_mut().def_insn("ldh_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x26",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_half(0) as i32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldh_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x26",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_half(0) as i64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldh_u32", BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x26",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_half(0) as u32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldh_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x26",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_half(0) as u64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val));
        }
    );
    itp.borrow_mut().def_insn("ldw_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x27",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_word(0) as i32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldw_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x27",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_word(0) as i64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldw_u32", BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x27",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_word(0) as u32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldw_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x27",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_word(0) as u64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val));
        }
    );
    itp.borrow_mut().def_insn("ldd_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x28",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Dword
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_dword(0) as i64;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldd_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x28",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Read Mem: Dword
            let val = proc0.mem_read((rs1 + rs2) as usize, 1).get_dword(0) as u64;
            // 4. Set rd(i64)
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(val));
        }
    );
    // Type:S 
    itp.borrow_mut().def_insn("stb_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x29",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i8)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i8(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i8(0)
            } else {
                0
            };
            // 3. Get imm(i32)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 4. Write Mem: Byte
            proc0.mem_write((rs1 + imm) as usize, Value::i8(rs2));
        }
    );
    itp.borrow_mut().def_insn("stb_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x29",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i8)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i8(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i8(0)
            } else {
                0
            };
            // 3. Get imm(i64)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 4. Write Mem: Byte
            proc0.mem_write((rs1 + imm) as usize, Value::i8(rs2));
        }
    );
    itp.borrow_mut().def_insn("sth_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x2a",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i16)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i16(0)
            } else {
                0
            };
            // 3. Get imm(i32)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 4. Write Mem: Half
            proc0.mem_write((rs1 + imm) as usize, Value::i16(rs2));
        }
    );
    itp.borrow_mut().def_insn("sth_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x2a",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i16)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i16(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i16(0)
            } else {
                0
            };
            // 3. Get imm(i64)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 4. Write Mem: Half
            proc0.mem_write((rs1 + imm) as usize, Value::i16(rs2));
        }
    );
    itp.borrow_mut().def_insn("stw_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x2b",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 3. Get imm(i32)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 4. Write Mem: Word
            proc0.mem_write((rs1 + imm) as usize, Value::i32(rs2));
        }
    );
    itp.borrow_mut().def_insn("stw_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x2b",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 3. Get imm(i64)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 4. Write Mem: Word
            proc0.mem_write((rs1 + imm) as usize, Value::i32(rs2));
        }
    );
    itp.borrow_mut().def_insn("std_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0x2c",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[0].sym() == OPR_REG {
                proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[0].sym() == OPR_IMM {
                insn.opr[0].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 3. Get imm(i64)
            let imm = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 4. Write Mem: DWord
            proc0.mem_write((rs1 + imm) as usize, Value::i64(rs2));
        }
    );

    // ============================================================================== //
    //                         Register Merge Instructions
    // ============================================================================== //

    itp.borrow_mut().def_insn("add2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x81",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] + [rs2_high: rs2_low] ======== //
            log_info!("TODO: add2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("add2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x81",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] + [rs2_high: rs2_low] ======== //
            log_info!("TODO: add2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("sub2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x82",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] - [rs2_high: rs2_low] ======== //
            log_info!("TODO: sub2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("sub2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x82",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] - [rs2_high: rs2_low] ======== //
            log_info!("TODO: sub2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x83",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x83",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x83",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], "E", "0x83",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );

    itp.borrow_mut().def_insn("etr2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_IMM], "E", "0x84",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr2_i32 rd, rs1, rs2, pos");
        }
    );
    itp.borrow_mut().def_insn("etr2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_IMM], "E", "0x84",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr2_i64 rd, rs1, rs2, pos");
        }
    );

    // ============================================================================== //
    //                               Control Instructions
    // ============================================================================== //

    itp.borrow_mut().def_insn("call" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG | OPR_IMM], "E", "0xe1",
        |cpu, insn| {
            // ======== call ======== //
            log_info!("TODO: call ptr");
        }
    );
    itp.borrow_mut().def_insn("label" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_LAB, OPR_REG | OPR_IMM], "E", "0xe2",
        |cpu, insn| {
            // ======== new label ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get label
            let mut lab = insn.opr[0].clone();
            println!("lab: {:?}", lab);
            // 2. Get val
            let val = if insn.opr[1].sym() == OPR_REG || insn.opr[1].sym() == OPR_IMM  {
                insn.opr[1].val()
            } else {
                Value::i64(0)
            };
            // 3. Set label
            lab.set_label(val);
            proc0.set_label(Rc::new(RefCell::new(lab)));
        }
    );
    itp.borrow_mut().def_insn("unlabel" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_LAB], "E", "0xe3",
        |cpu, insn| {
            // ======== unlabel ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get label
            let lab = insn.opr[0].clone();
            println!("lab: {:?}", lab);
            // 2. Del label
            proc0.del_label(lab.label_nick());
        }
    );
    itp.borrow_mut().def_insn("cond_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0xe4",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i32(0)
            } else {
                0
            };
            // 2. Get rs2(i32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i32(0)
            } else {
                0
            };
            // 3. Get cc
            let cc = if insn.opr[3].sym() == OPR_REG {
                proc0.get_reg(insn.opr[3].val().get_byte(0) as usize).get_i32(0)
            } else if insn.opr[3].sym() == OPR_IMM {
                insn.opr[3].val().get_i32(0)
            } else {
                0
            };
            // 4. Get res
            let res = match cc as u8 {
                COND_NO => false,
                COND_AL => true,
                COND_EQ => rs1 == rs2,
                COND_NE => rs1 != rs2,
                COND_LT => rs1 < rs2,
                COND_LE => rs1 <= rs2,
                COND_GT => rs1 > rs2,
                COND_GE => rs1 >= rs2,
                _ => false
            };
            // 5. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("cond_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0xe4", 
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2) ======== //
            // 1. Get rs1(i64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                cpu.proc.borrow().get_reg(insn.opr[1].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_i64(0)
            } else {
                0
            };
            // 2. Get rs2(i64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                cpu.proc.borrow().get_reg(insn.opr[2].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_i64(0)
            } else {
                0
            };
            // 3. Get cc
            let cc = if insn.opr[3].sym() == OPR_REG {
                cpu.proc.borrow().get_reg(insn.opr[3].val().get_byte(0) as usize).get_i64(0)
            } else if insn.opr[3].sym() == OPR_IMM {
                insn.opr[3].val().get_i64(0)
            } else {
                0
            };
            // 4. Get res
            let res = match cc as u8 {
                COND_NO => false,
                COND_AL => true,
                COND_EQ => rs1 == rs2,
                COND_NE => rs1 != rs2,
                COND_LT => rs1 < rs2,
                COND_LE => rs1 <= rs2,
                COND_GT => rs1 > rs2,
                COND_GE => rs1 >= rs2,
                _ => false
            };
            // 5. Set rd
            cpu.proc.borrow_mut().set_reg(insn.opr[0].val().get_byte(0) as usize, Value::i64(res as i64));
        }   
    );
    itp.borrow_mut().def_insn("cond_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0xe4",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u32(0)
            } else {
                0
            };
            // 2. Get rs2(u32)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u32(0)
            } else {
                0
            };
            // 3. Get cc
            let cc = if insn.opr[3].sym() == OPR_REG {
                proc0.get_reg(insn.opr[3].val().get_byte(0) as usize).get_u32(0)
            } else if insn.opr[3].sym() == OPR_IMM {
                insn.opr[3].val().get_u32(0)
            } else {
                0
            };
            // 4. Get res
            let res = match cc as u8 {
                COND_NO => false,
                COND_AL => true,
                COND_EQ => rs1 == rs2,
                COND_NE => rs1 != rs2,
                COND_LT => rs1 < rs2,
                COND_LE => rs1 <= rs2,
                COND_GT => rs1 > rs2,
                COND_GE => rs1 >= rs2,
                _ => false
            };
            // 5. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u32(res as u32));
        }
    );
    itp.borrow_mut().def_insn("cond_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM, OPR_REG | OPR_IMM], "E", "0xe4",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = if insn.opr[1].sym() == OPR_REG {
                proc0.get_reg(insn.opr[1].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[1].sym() == OPR_IMM {
                insn.opr[1].val().get_u64(0)
            } else {
                0
            };
            // 2. Get rs2(u64)
            let rs2 = if insn.opr[2].sym() == OPR_REG {
                proc0.get_reg(insn.opr[2].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[2].sym() == OPR_IMM {
                insn.opr[2].val().get_u64(0)
            } else {
                0
            };
            // 3. Get cc
            let cc = if insn.opr[3].sym() == OPR_REG {
                proc0.get_reg(insn.opr[3].val().get_byte(0) as usize).get_u64(0)
            } else if insn.opr[3].sym() == OPR_IMM {
                insn.opr[3].val().get_u64(0)
            } else {
                0
            };
            // 4. Get res
            let res = match cc as u8 {
                COND_NO => false,
                COND_AL => true,
                COND_EQ => rs1 == rs2,
                COND_NE => rs1 != rs2,
                COND_LT => rs1 < rs2,
                COND_LE => rs1 <= rs2,
                COND_GT => rs1 > rs2,
                COND_GE => rs1 >= rs2,
                _ => false
            };
            // 5. Set rd
            proc0.set_reg(insn.opr[0].val().get_byte(0) as usize, Value::u64(res as u64));
        }
    );

    itp.borrow_mut().def_insn("cond2" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_IMM], "E", "0xe5",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2) ======== //
            log_info!("TODO: cond2 rd, rs1_low, rs1_high, rs2_low, rs2_high, cc");
        }
    );
    itp.borrow_mut().def_insn("condval" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM, OPR_IMM], "E", "0xe6",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2 ? v1 : v2) ======== //
            log_info!("TODO: condval rd, rs1, rs2, cc, v1, v2");
        }
    );
    itp.borrow_mut().def_insn("jmp" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_LAB], "E", "0xe7",
        |cpu, insn| {
            // ======== jump ======== //
            log_info!("TODO: jmp $label");
        }
    );
    itp.borrow_mut().def_insn("brcond" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_LAB, OPR_REG, OPR_REG, OPR_IMM], "E", "0xe8",
        |cpu, insn| {
            // ======== brcond ======== //
            log_info!("TODO: brcond $label, rs1, rs2, cc");
        }
    );
    itp.borrow_mut().def_insn("brcond2" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_LAB, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_IMM], "E", "0xe9",
        |cpu, insn| {
            // ======== brcond ======== //
            log_info!("TODO: brc
                0ond2 $label, rs1_low, rs1_high, rs2_low, rs2_high, cc");
        }
    );
    itp.borrow_mut().def_insn("goto_tb" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_IMM], "E", "0xea",
        |cpu, insn| {
            // ======== exit tb and return 0 to rd ======== //
            log_info!("TODO: goto_tb rd");
        }
    );
    itp.borrow_mut().def_insn("exit_tb" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_IMM, OPR_IMM], "E", "0xeb",
        |cpu, insn| {
            // ======== exit tb and return 0 to rd ======== //
            log_info!("TODO: exit_tb rd");
        }
    );
    itp.borrow_mut().def_insn("trap" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_IMM], "E", "0xec",
        |cpu, insn| {
            // ======== trap TRAP_CODE ======== //
            log_info!("TODO: trap TRAP_CODE");
        }
    );
    itp.borrow_mut().def_insn("yes" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_OFF | OPR_IMM | OPR_REG], "E", "0xff",
        |cpu, insn| {
            let proc0 = cpu.proc.borrow().clone();
            let val = if insn.opr[0].sym() == OPR_REG {
                Some(proc0.get_reg(insn.opr[0].val().get_byte(0) as usize).get_u64(0))
            } else if insn.opr[0].sym() == OPR_IMM {
                Some(insn.opr[0].val().get_dword(0))
            } else {
                None
            };
            println!("yes yes yes: {} !!!", if val.is_some() { val.unwrap().to_string() } else { "".to_string() });
        }
    );



    Some(itp)
}



pub fn evo_encode(insn: &mut Instruction, opr: Vec<Operand>) -> Instruction {
    if opr.len() == 0 {
        let mut res = insn.clone();
        res.is_applied = true;
        return res;
    }
    let mut new_opr = opr.clone();
    let mut is_applied = true;
    // Check syms
    if insn.check_syms(opr.clone()) {
        // match opcode type kind and fill bytes by opreands
        let mut code: Vec<u8> = vec![];
        match insn.opc.kind() {
            OpcodeKind::E(_, _) => {
                // 1. deal with opcode
                let opcode = insn.code.get_bin(0);
                let flag_sb: u8 = match (insn.is_unsigned(), insn.is_64bit()) {
                    (false, false) => 0b00,
                    (false, true) => 0b01,
                    (true, false) => 0b10,
                    (true, true) => 0b11,
                };
                if opcode.len() == 0 {
                    // 1.1 No opcode
                    log_error!("Encode opcode failed: {} , void opcode", insn.opc.name());
                } else if opcode.len() == 1 {
                    // 1.2 Single byte opcode
                    code = vec![flag_sb, opcode[0]];
                    let syms = opr.iter().map(|x| x.sym()).collect::<Vec<_>>();
                    match (syms.as_slice(), insn.is_64bit()) {
                        // Match Most Format in A field
                        ([], _) => {
                            code[0] = 0b000_000_00 | flag_sb;
                        },
                        ([OPR_LAB], _) => {
                            code[0] = 0b000_001_00 | flag_sb;
                            let lab = &opr[0];
                            if lab.label_addr().scale_sum() > 32 {
                                code.extend_from_slice(&lab.val().get_dword(0).to_le_bytes());
                            } else {
                                code.extend_from_slice(&lab.val().get_word(0).to_le_bytes());
                            }
                        },
                        ([OPR_LAB, OPR_REG], _) => {
                            code[0] = 0b000_001_00 | flag_sb;
                            // Get reg idx from opr
                            let reg1 = &opr[1];
                            code.push(reg1.val().get_byte(1));
                        },
                        ([OPR_REG], _) => {
                            code[0] = 0b000_001_00 | flag_sb;
                            // Get reg idx from opr
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                        },
                        ([OPR_REG, OPR_REG], _) => {
                            code[0] = 0b000_010_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                        },
                        ([OPR_REG, OPR_REG, OPR_REG], _) => {
                            code[0] = 0b000_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_REG], _) => {
                            code[0] = 0b000_100_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let reg4 = &opr[3];
                            code.push(reg4.val().get_byte(0));
                        },
                        ([OPR_REG, OPR_REG, OPR_IMM], false) => {
                            code[0] = 0b000_101_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let imm1 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_IMM], true)  => {
                            code[0] = 0b000_101_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let imm1 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_IMM, OPR_IMM], false) => {
                            code[0] = 0b000_110_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let imm1 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            let imm2 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u32(imm2.val().get_word(0)));
                            code.extend_from_slice(&imm2.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_IMM, OPR_IMM], true)  => {
                            code[0] = 0b000_110_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let imm1 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            let imm2 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u64(imm2.val().get_dword(0)));
                            code.extend_from_slice(&imm2.val().get_dword(0).to_le_bytes());
                        },
                        // Match Most Format in A & B field
                        ([OPR_OFF], _) => {
                            code[0] = 0b000_000_00 | flag_sb;
                        },
                        ([OPR_IMM], false) => {
                            code[0] = 0b001_000_00 | flag_sb;
                            let imm1 = &opr[0];
                            new_opr[0] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_IMM], true) => {
                            code[0] = 0b001_000_00 | flag_sb;
                            let imm1 = &opr[0];
                            new_opr[0] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_LAB, OPR_IMM], _) => {
                            code[0] = 0b001_000_00 | flag_sb;
                            let imm1 = &opr[1];
                            if imm1.val().scale_sum() > 32{
                                code[0] |= 0b0000_0001;
                                new_opr[1] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                                code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            } else {
                                new_opr[1] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                                code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            }
                        },
                        ([OPR_IMM, OPR_IMM, OPR_IMM], false) => {
                            code[0] = 0b011_000_00 | flag_sb;
                            let imm1 = &opr[0];
                            new_opr[0] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            let imm2 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u32(imm2.val().get_word(0)));
                            code.extend_from_slice(&imm2.val().get_word(0).to_le_bytes());
                            let imm3 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u32(imm3.val().get_word(0)));
                            code.extend_from_slice(&imm3.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_IMM, OPR_IMM, OPR_IMM], true)  => {
                            code[0] = 0b011_000_00 | flag_sb;
                            let imm1 = &opr[0];
                            new_opr[0] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            let imm2 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u64(imm2.val().get_dword(0)));
                            code.extend_from_slice(&imm2.val().get_dword(0).to_le_bytes());
                            let imm3 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u64(imm3.val().get_dword(0)));
                            code.extend_from_slice(&imm3.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_IMM], false) => {
                            code[0] = 0b001_001_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let imm1 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_IMM], true)  => {
                            code[0] = 0b001_001_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let imm1 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_IMM, OPR_IMM], false) => {
                            code[0] = 0b010_001_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let imm1 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            let imm2 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u32(imm2.val().get_word(0)));
                            code.extend_from_slice(&imm2.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_IMM, OPR_IMM], true)  => {
                            code[0] = 0b010_001_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let imm1 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            let imm2 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u64(imm2.val().get_dword(0)));
                            code.extend_from_slice(&imm2.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_IMM, OPR_IMM, OPR_IMM], false) => {
                            code[0] = 0b011_001_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let imm1 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            let imm2 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u32(imm2.val().get_word(0)));
                            code.extend_from_slice(&imm2.val().get_word(0).to_le_bytes());
                            let imm3 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u32(imm3.val().get_word(0)));
                            code.extend_from_slice(&imm3.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_IMM, OPR_IMM, OPR_IMM], true)  => {
                            code[0] = 0b011_001_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let imm1 = &opr[1];
                            new_opr[1] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            let imm2 = &opr[2];
                            new_opr[2] = Operand::imm(Value::u64(imm2.val().get_dword(0)));
                            code.extend_from_slice(&imm2.val().get_dword(0).to_le_bytes());
                            let imm3 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u64(imm3.val().get_dword(0)));
                            code.extend_from_slice(&imm3.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_IMM], false) => {
                            code[0] = 0b001_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let imm1 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_IMM], true)  => {
                            code[0] = 0b001_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let imm1 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM], false) => {
                            code[0] = 0b001_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let imm1 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            let imm2 = &opr[4];
                            new_opr[4] = Operand::imm(Value::u32(imm2.val().get_word(0)));
                            code.extend_from_slice(&imm2.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM], true)  => {
                            code[0] = 0b001_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let imm1 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            let imm2 = &opr[4];
                            new_opr[4] = Operand::imm(Value::u64(imm2.val().get_dword(0)));
                            code.extend_from_slice(&imm2.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM, OPR_IMM], false) => {
                            code[0] = 0b011_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let imm1 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                            let imm2 = &opr[4];
                            new_opr[4] = Operand::imm(Value::u32(imm2.val().get_word(0)));
                            code.extend_from_slice(&imm2.val().get_word(0).to_le_bytes());
                            let imm3 = &opr[5];
                            new_opr[5] = Operand::imm(Value::u32(imm3.val().get_word(0)));
                            code.extend_from_slice(&imm3.val().get_word(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_IMM, OPR_IMM, OPR_IMM], true)  => {
                            code[0] = 0b011_011_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let imm1 = &opr[3];
                            new_opr[3] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                            let imm2 = &opr[4];
                            new_opr[4] = Operand::imm(Value::u64(imm2.val().get_dword(0)));
                            code.extend_from_slice(&imm2.val().get_dword(0).to_le_bytes());
                            let imm3 = &opr[5];
                            new_opr[5] = Operand::imm(Value::u64(imm3.val().get_dword(0)));
                            code.extend_from_slice(&imm3.val().get_dword(0).to_le_bytes());
                        },
                        // Extend field
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_IMM], false) => {
                            // A & B field
                            code[0] = 0b110_100_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let reg4 = &opr[3];
                            code.push(reg4.val().get_byte(0));
                            // ExtC
                            code.push(0b001_001_00);
                            let reg5 = &opr[4];
                            code.push(reg5.val().get_byte(0));
                            let imm1 = &opr[5];
                            new_opr[4] = Operand::imm(Value::u32(imm1.val().get_word(0)));
                            code.extend_from_slice(&imm1.val().get_word(0).to_le_bytes());
                        }
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_IMM], true)  => {
                            // A & B field
                            code[0] = 0b110_100_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let reg4 = &opr[3];
                            code.push(reg4.val().get_byte(0));
                            // ExtC
                            code.push(0b001_001_00);
                            let reg5 = &opr[4];
                            code.push(reg5.val().get_byte(0));
                            let imm1 = &opr[5];
                            new_opr[4] = Operand::imm(Value::u64(imm1.val().get_dword(0)));
                            code.extend_from_slice(&imm1.val().get_dword(0).to_le_bytes());
                        },
                        ([OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG, OPR_REG], _) => {
                            // A & B field
                            code[0] = 0b110_100_00 | flag_sb;
                            let reg1 = &opr[0];
                            code.push(reg1.val().get_byte(0));
                            let reg2 = &opr[1];
                            code.push(reg2.val().get_byte(0));
                            let reg3 = &opr[2];
                            code.push(reg3.val().get_byte(0));
                            let reg4 = &opr[3];
                            code.push(reg4.val().get_byte(0));
                            // ExtC
                            code.push(0b000_010_00);
                            let reg5 = &opr[4];
                            code.push(reg5.val().get_byte(0));
                            let reg6 = &opr[5];
                            code.push(reg6.val().get_byte(0));
                        },
                        _ => {
                            is_applied = false;
                            log_error!("Encode operands not implemented: {} {}", insn.opc.name(), syms.iter().map(|x| OperandKind::sym_str(*x)).collect::<Vec<_>>().join(", "));
                        }
                    };
                } else if opcode.len() == 2 {
                    // 1.3 Double byte opcode
                    code = vec![flag_sb | 0b000_111_00, opcode[0], opcode[1]];
                }
            },
            _ => {
                is_applied = false;
                log_error!("Not support opcode type {} in arch {}", insn.opc.kind(), EVO_ARCH);
            }
        }
        // refresh status
        let mut res = insn.clone();
        res.opr = new_opr;
        res.is_applied = is_applied;
        res.code = Value::array_u8(RefCell::new(code));
        res
    } else {
        log_error!("Encode operands failed: {} , check syms: {}", insn.opc.name(), opr.iter().map(|x| OperandKind::sym_str(x.sym())).collect::<Vec<_>>().join(", "));
        let mut res = insn.clone();
        res.is_applied = false;
        res.opr = opr.clone();
        // Error
        res
    }
}


/// decode from Value
pub fn evo_decode(value: Value) -> Instruction {
    let mut res = Instruction::undef();
    // 1. check scale
    if value.size() < 2 {
        log_error!("Decode fail: invalid insn scale {}", value.scale_sum());
        return res;
    }
    // 2. get flag & opc
    res.set_arch(&EVO_ARCH);
    res.code = value;
    let flag_prefix = res.code.get_byte(0);
    let is_unsigned = (flag_prefix & 0b000_000_10) == 0b000_000_10;
    let is_64bit = (flag_prefix & 0b000_000_01) == 0b000_000_01;
    let flag_a = (flag_prefix & 0b000_111_00) >> 2;
    let flag_b = (flag_prefix & 0b111_000_00) >> 5;
    let mut opr = vec![];
    let opcode;
    let mut cur_idx: usize;

    if flag_a == 0b111 {
        // 2 opcodes
        opcode = vec![res.code.get_byte(1), res.code.get_byte(2)];
        cur_idx = 3;
    } else {
        // 1 opcodes
        opcode = vec![res.code.get_byte(1)];
        cur_idx = 2;
    }

    // 3. match opcode
    if opcode.len() == 1 {
        match (opcode[0], is_unsigned, is_64bit) {
            (0x01, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "add_i32").borrow().clone(),
            (0x01, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "add_i64").borrow().clone(),
            (0x02, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "sub_i32").borrow().clone(),
            (0x02, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "sub_i64").borrow().clone(),
            (0x03, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "neg_i32").borrow().clone(),
            (0x03, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "neg_i64").borrow().clone(),
            (0x04, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "mul_i32").borrow().clone(),
            (0x04, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "mul_i64").borrow().clone(),
            (0x05, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "div_i32").borrow().clone(),  
            (0x05, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "div_i64").borrow().clone(),
            (0x05, true , false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "div_u32").borrow().clone(),
            (0x05, true , true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "div_u64").borrow().clone(),
            (0x06, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "rem_i32").borrow().clone(),
            (0x06, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "rem_i64").borrow().clone(),
            (0x06, true , false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "rem_u32").borrow().clone(),
            (0x06, true , true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "rem_u64").borrow().clone(),

            (0x07, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "and_i32").borrow().clone(),
            (0x07, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "and_i64").borrow().clone(),
            (0x08, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "or_i32").borrow().clone(),
            (0x08, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "or_i64").borrow().clone(),
            (0x09, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "xor_i32").borrow().clone(),
            (0x09, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "xor_i64").borrow().clone(),
            (0x0a, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "not_i32").borrow().clone(),
            (0x0a, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "not_i64").borrow().clone(),
            (0x0b, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "andc_i32").borrow().clone(),
            (0x0b, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "andc_i64").borrow().clone(),
            (0x0c, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "eqv_i32").borrow().clone(),
            (0x0c, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "eqv_i64").borrow().clone(),
            (0x0d, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "nand_i32").borrow().clone(),
            (0x0d, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "nand_i64").borrow().clone(),
            (0x0e, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "nor_i32").borrow().clone(),
            (0x0e, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "nor_i64").borrow().clone(),
            (0x0f, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "orc_i32").borrow().clone(),
            (0x0f, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "orc_i64").borrow().clone(),

            (0x10, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "clz_i32").borrow().clone(),
            (0x10, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "clz_i64").borrow().clone(),
            (0x11, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "ctz_i32").borrow().clone(),
            (0x11, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "ctz_i64").borrow().clone(),
            (0x12, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "shl_i32").borrow().clone(),
            (0x12, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "shl_i64").borrow().clone(),
            (0x13, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "shr_i32").borrow().clone(),
            (0x13, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "shr_i64").borrow().clone(),
            (0x14, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "sar_i32").borrow().clone(),
            (0x14, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "sar_i64").borrow().clone(),
            (0x15, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "rol_i32").borrow().clone(),
            (0x15, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "rol_i64").borrow().clone(),
            (0x16, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "ror_i32").borrow().clone(),
            (0x16, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "ror_i64").borrow().clone(),

            (0x17, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "mov_i32").borrow().clone(),
            (0x17, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "mov_i64").borrow().clone(),
            (0x18, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extb_i32").borrow().clone(),
            (0x18, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extb_i64").borrow().clone(),
            (0x18, true, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extb_u32").borrow().clone(),
            (0x18, true, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extb_u64").borrow().clone(),
            (0x19, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "exth_i32").borrow().clone(),
            (0x19, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "exth_i64").borrow().clone(),
            (0x19, true, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "exth_u32").borrow().clone(),
            (0x19, true, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "exth_u64").borrow().clone(),
            (0x1a, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extw_i32").borrow().clone(),
            (0x1a, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extw_i64").borrow().clone(),
            (0x1a, true, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extw_u32").borrow().clone(),
            (0x1a, true, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extw_u64").borrow().clone(),

            (0x1b, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "etr_i32").borrow().clone(),
            (0x1b, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "etr_i64").borrow().clone(),
            (0x1b, true, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "etr_u32").borrow().clone(),
            (0x1b, true, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "etr_u64").borrow().clone(),
            (0x1c, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extl_i32").borrow().clone(),
            (0x1c, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "extl_i64").borrow().clone(),
            (0x1d, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "exth_i32").borrow().clone(),
            (0x1d, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "exth_i64").borrow().clone(),

            (0x1e, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "truc_i32").borrow().clone(),
            (0x1e, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "truc_i64").borrow().clone(),
            (0x1f, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "conc_i64").borrow().clone(),
            (0x20, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "conn_i64").borrow().clone(),
            (0x21, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "swph_i32").borrow().clone(),
            (0x21, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "swph_i64").borrow().clone(),
            (0x22, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "swpw_i32").borrow().clone(),
            (0x22, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "swpw_i64").borrow().clone(),
            (0x23, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "swpd_i64").borrow().clone(),
            (0x24, false, false) => res = Instruction::insn_pool_nget(&EVO_ARCH, "depo_i32").borrow().clone(),
            (0x24, false, true) => res = Instruction::insn_pool_nget(&EVO_ARCH, "depo_i64").borrow().clone(),

            _ => {
                log_error!("Decode opcode fail: 0x{:x}", opcode[0]);
            }
        };

        // 4. deal with operands
        match(flag_a, is_64bit) {
            (0b000, _) => cur_idx,
            (0b001, _) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                cur_idx
            },
            (0b010, _) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                cur_idx
            },
            (0b011, _) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                cur_idx
            },
            (0b100, _) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                cur_idx
            },
            (0b101, false) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                cur_idx
            },
            (0b101, true) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                cur_idx
            },
            (0b110, false) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                cur_idx
            },
            (0b110, true) => {
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.code.get_byte(cur_idx) as usize).borrow().clone());
                cur_idx += 1;
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                cur_idx
            },
            _ => {
                log_error!("Decode operand fail: 0x{:x}", res.code.get_byte(cur_idx));
                cur_idx
            }
        };

        match(flag_b, is_64bit) {
            (0b000, _) => cur_idx,
            (0b001, false) => {
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                cur_idx
            },
            (0b001, true) => {
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                cur_idx
            },
            (0b010, false) => {
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                cur_idx
            },
            (0b010, true) => {
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                cur_idx
            },
            (0b011, false) => {
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                opr.push(Operand::imm(Value::i32(res.code.get_word(cur_idx) as i32)));
                cur_idx += 4;
                cur_idx
            },
            (0b011, true) => {
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                opr.push(Operand::imm(Value::i64(res.code.get_dword(cur_idx) as i64)));
                cur_idx += 8;
                cur_idx
            },
            (0b100, false) => {
                let base = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let idx = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let scale = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let disp = res.code.get_word(cur_idx) as i32;
                cur_idx += 4;
                opr.push(Operand::mem(
                    RegFile::reg_poolr_get(&EVO_ARCH, base as usize).borrow().clone(), 
                    RegFile::reg_poolr_get(&EVO_ARCH, idx as usize).borrow().clone(),
                    Value::u8(scale), 
                    Value::i32(disp))
                );
                cur_idx
            },
            (0b100, true) => {
                let base = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let idx = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let scale = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let disp = res.code.get_dword(cur_idx) as i64;
                cur_idx += 8;
                opr.push(Operand::mem(
                    RegFile::reg_poolr_get(&EVO_ARCH, base as usize).borrow().clone(), 
                    RegFile::reg_poolr_get(&EVO_ARCH, idx as usize).borrow().clone(),
                    Value::u8(scale),
                    Value::i64(disp))
                );
                cur_idx
            },
            (0b101, false) => {
                let base = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let idx = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let scale = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let disp = res.code.get_word(cur_idx) as i32;
                cur_idx += 4;
                opr.push(Operand::mem(
                    RegFile::reg_poolr_get(&EVO_ARCH, base as usize).borrow().clone(), 
                    RegFile::reg_poolr_get(&EVO_ARCH, idx as usize).borrow().clone(),
                    Value::u8(scale), 
                    Value::i32(disp))
                );
                let base = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let idx = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let scale = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let disp = res.code.get_word(cur_idx) as i32;
                cur_idx += 4;
                opr.push(Operand::mem(
                    RegFile::reg_poolr_get(&EVO_ARCH, base as usize).borrow().clone(), 
                    RegFile::reg_poolr_get(&EVO_ARCH, idx as usize).borrow().clone(),
                    Value::u8(scale), 
                    Value::i32(disp))
                );
                cur_idx
            },
            (0b101, true) => {
                let base = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let idx = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let scale = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let disp = res.code.get_dword(cur_idx) as i64;
                cur_idx += 8;
                opr.push(Operand::mem(
                    RegFile::reg_poolr_get(&EVO_ARCH, base as usize).borrow().clone(), 
                    RegFile::reg_poolr_get(&EVO_ARCH, idx as usize).borrow().clone(),
                    Value::u8(scale),
                    Value::i64(disp))
                );
                let base = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let idx = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let scale = res.code.get_byte(cur_idx);
                cur_idx += 1;
                let disp = res.code.get_dword(cur_idx) as i64;
                cur_idx += 8;
                opr.push(Operand::mem(
                    RegFile::reg_poolr_get(&EVO_ARCH, base as usize).borrow().clone(), 
                    RegFile::reg_poolr_get(&EVO_ARCH, idx as usize).borrow().clone(),
                    Value::u8(scale),
                    Value::i64(disp))
                );
                cur_idx
            },
            (0b110, false) => {
                cur_idx
            },
            (0b110, true) => {
                cur_idx
            },
            _ => {
                log_error!("Decode operand fail: 0x{:x}", res.code.get_byte(cur_idx));
                cur_idx
            }
        };

    } else if opcode.len() == 2 {

    }
    

    // 3. encode
    res.encode(opr)
}

#[macro_export]
macro_rules! evo_gen {
    ($evo_opcode:literal $(, $($evo_operands:expr),*)?) => {
        Instruction::insn_pool_nget(&EVO_ARCH, $evo_opcode).borrow().clone().encode(vec![
            $($($evo_operands,)*)?
        ])
    };
}


#[cfg(test)]
mod evo_test {

    use super::*;
    use crate::core::cpu::CPUState;


    #[test]
    fn evo_itp() {
        let cpu = CPUState::init(&EVO_ARCH, &EVO_ARCH, None, None, None);
        cpu.set_nreg("t1", Value::i64(266));
        cpu.set_nreg("t2", Value::i64(17));
        cpu.set_nreg("t3", Value::i64(65535));
        cpu.set_nreg("t4", Value::i64(65537));
        cpu.set_nreg("t5", Value::f64(3.1415926));
        cpu.mem_write(26, Value::i32(0x1ffff));
        println!("{}", cpu.pool_info());

        // R-Type Insns Test 
        let mut insn1  = Instruction::from_string(&EVO_ARCH, "add_i32 t0, t1, t1");
        let insn2  = Instruction::from_string(&EVO_ARCH, "add_i64 t0, t2, 65535");
        let insn3  = Instruction::from_string(&EVO_ARCH, "sub_i32 t0, 4322, -4321");
        let insn4  = Instruction::from_string(&EVO_ARCH, "sub_i64 t0, t1, t2");
        let insn5  = Instruction::from_string(&EVO_ARCH, "neg_i32 t0, t1");
        let insn6  = Instruction::from_string(&EVO_ARCH, "neg_i64 t0, t1");
        let insn7  = Instruction::from_string(&EVO_ARCH, "mul_i32 t0, t1, t2");
        let insn8  = Instruction::from_string(&EVO_ARCH, "mul_i64 t0, t1, t2");
        let insn9  = Instruction::from_string(&EVO_ARCH, "div_i32 t0, t1, t2");
        let insn10 = Instruction::from_string(&EVO_ARCH, "div_i64 t0, t1, t2");
        let insn11 = Instruction::from_string(&EVO_ARCH, "div_u32 t0, t1, t2");
        let insn12 = Instruction::from_string(&EVO_ARCH, "div_u64 t0, t1, t2");
        let insn13 = Instruction::from_string(&EVO_ARCH, "rem_i32 t0, t1, t2");
        let insn14 = Instruction::from_string(&EVO_ARCH, "rem_i64 t0, t1, t2");
        let insn15 = Instruction::from_string(&EVO_ARCH, "rem_u32 t0, t1, t2");
        let insn16 = Instruction::from_string(&EVO_ARCH, "rem_u64 t0, t1, t2");

        let insn17 = Instruction::from_string(&EVO_ARCH, "and_i32 t0, t1, t2");
        let insn18 = Instruction::from_string(&EVO_ARCH, "and_i64 t0, t1, t2");
        let insn19 = Instruction::from_string(&EVO_ARCH, "or_i32 t0, t1, t2");
        let insn20 = Instruction::from_string(&EVO_ARCH, "or_i64 t0, t1, t2");
        let insn21 = Instruction::from_string(&EVO_ARCH, "xor_i32 t0, t1, t2");
        let insn22 = Instruction::from_string(&EVO_ARCH, "xor_i64 t0, t1, t2");
        let insn23 = Instruction::from_string(&EVO_ARCH, "not_i32 t0, t1");
        let insn24 = Instruction::from_string(&EVO_ARCH, "not_i64 t0, t1");
        let insn25 = Instruction::from_string(&EVO_ARCH, "andc_i32 t0, t1, t2");
        let insn26 = Instruction::from_string(&EVO_ARCH, "andc_i64 t0, t1, t2");
        let insn27 = Instruction::from_string(&EVO_ARCH, "eqv_i32 t0, t1, t2");
        let insn28 = Instruction::from_string(&EVO_ARCH, "eqv_i64 t0, t1, t2");
        let insn29 = Instruction::from_string(&EVO_ARCH, "nand_i32 t0, t1, t2");
        let insn30 = Instruction::from_string(&EVO_ARCH, "nand_i64 t0, t1, t2");
        let insn31 = Instruction::from_string(&EVO_ARCH, "nor_i32 t0, t1, t2");
        let insn32 = Instruction::from_string(&EVO_ARCH, "nor_i64 t0, t1, t2");
        let insn33 = Instruction::from_string(&EVO_ARCH, "orc_i32 t0, t1, t2");
        let insn34 = Instruction::from_string(&EVO_ARCH, "orc_i64 t0, t1, t2");

        let insn35 = Instruction::from_string(&EVO_ARCH, "clz_i32 t0, t1, t2");
        let insn36 = Instruction::from_string(&EVO_ARCH, "clz_i64 t0, t1, t2");
        let insn37 = Instruction::from_string(&EVO_ARCH, "ctz_i32 t0, t1, t2");
        let insn38 = Instruction::from_string(&EVO_ARCH, "ctz_i64 t0, t1, t2");
        let insn39 = Instruction::from_string(&EVO_ARCH, "shl_i32 t0, t1, t2");
        let insn40 = Instruction::from_string(&EVO_ARCH, "shl_i64 t0, t1, t2");
        let insn41 = Instruction::from_string(&EVO_ARCH, "shr_i32 t0, t1, t2");
        let insn42 = Instruction::from_string(&EVO_ARCH, "shr_i64 t0, t1, t2");
        let insn43 = Instruction::from_string(&EVO_ARCH, "sar_i32 t0, t1, t2");
        let insn44 = Instruction::from_string(&EVO_ARCH, "sar_i64 t0, t1, t2");
        let insn45 = Instruction::from_string(&EVO_ARCH, "rol_i32 t0, t1, t2");
        let insn46 = Instruction::from_string(&EVO_ARCH, "rol_i64 t0, t1, t2");
        let insn47 = Instruction::from_string(&EVO_ARCH, "ror_i32 t0, t1, t2");
        let insn48 = Instruction::from_string(&EVO_ARCH, "ror_i64 t0, t1, t2");

        let insn49 = Instruction::from_string(&EVO_ARCH, "mov_i32 t0, t1");
        let insn50 = Instruction::from_string(&EVO_ARCH, "mov_i64 t0, t1");
        let insn51 = Instruction::from_string(&EVO_ARCH, "extb_i32 t0, t3");
        let insn52 = Instruction::from_string(&EVO_ARCH, "extb_i64 t0, t3");
        let insn53 = Instruction::from_string(&EVO_ARCH, "extb_u32 t0, t3");
        let insn54 = Instruction::from_string(&EVO_ARCH, "extb_u64 t0, t3");
        let insn55 = Instruction::from_string(&EVO_ARCH, "exth_i32 t0, 0x32 41 e3");
        let insn56 = Instruction::from_string(&EVO_ARCH, "exth_i64 t0, t3");
        let insn57 = Instruction::from_string(&EVO_ARCH, "exth_u32 t0, t3");
        let insn58 = Instruction::from_string(&EVO_ARCH, "exth_u64 t0, t3");
        let insn59 = Instruction::from_string(&EVO_ARCH, "extw_i32 t0, 0xf2");
        let insn60 = Instruction::from_string(&EVO_ARCH, "extw_i64 t0, t3");
        let insn61 = Instruction::from_string(&EVO_ARCH, "extw_u32 t0, t3");
        let insn62 = Instruction::from_string(&EVO_ARCH, "extw_u64 t0, t3");


        let insn63 = Instruction::from_string(&EVO_ARCH, "etr_i32 t0, t4, 1, 3");
        let insn64 = Instruction::from_string(&EVO_ARCH, "etr_i64 t0, t4, 3, 12");
        let insn65 = Instruction::from_string(&EVO_ARCH, "etr_u32 t0, t4, 0, 4");
        let insn66 = Instruction::from_string(&EVO_ARCH, "etr_u64 t0, t4, 0, 8");
        let insn67 = Instruction::from_string(&EVO_ARCH, "etrh_i32 t0, t4");
        let insn68 = Instruction::from_string(&EVO_ARCH, "etrl_i32 t0, t4");
        let insn69 = Instruction::from_string(&EVO_ARCH, "etrh_i64 t0, t4");
        let insn70 = Instruction::from_string(&EVO_ARCH, "etrl_i64 t0, t4");

        let insn71 = Instruction::from_string(&EVO_ARCH, "truc_i32 t0, t4");
        let insn72 = Instruction::from_string(&EVO_ARCH, "truc_i64 t0, t4");
        let insn73 = Instruction::from_string(&EVO_ARCH, "conc_i64 t0, t4, t1");
        let insn74 = Instruction::from_string(&EVO_ARCH, "conn_i64 t0, t4, t1");
        let insn75 = Instruction::from_string(&EVO_ARCH, "swph_i32 t0, t4");
        let insn76 = Instruction::from_string(&EVO_ARCH, "swph_i64 t0, t4");
        let insn77 = Instruction::from_string(&EVO_ARCH, "swpw_i32 t0, t4");
        let insn78 = Instruction::from_string(&EVO_ARCH, "swpw_i64 t0, t4");
        let insn79 = Instruction::from_string(&EVO_ARCH, "swpd_i64 t0, t4");
        let insn80 = Instruction::from_string(&EVO_ARCH, "depo_i32 t0, t4");
        let insn81 = Instruction::from_string(&EVO_ARCH, "depo_i64 t0, t4");
        let insn82 = Instruction::from_string(&EVO_ARCH, "label xss, 0x23 4e ff 8a 23");
        let insn83 = Instruction::from_string(&EVO_ARCH, "unlabel xss, 0x12");
        let insn84 = Instruction::from_string(&EVO_ARCH, "yes");

        insn1.set_label(Some("sieve".to_string()));
        cpu.execute(&insn1);
        println!("{:<60} {:<70} -> t0 = {}", insn1.code.hex(0, -1, false), insn1.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn2);
        println!("{:<60} {:<70} -> t0 = {}", insn2.code.hex(0, -1, false), insn2.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn3);
        println!("{:<60} {:<70} -> t0 = {}", insn3.code.hex(0, -1, false), insn3.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn4);
        println!("{:<60} {:<70} -> t0 = {}", insn4.code.hex(0, -1, false), insn4.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn5);
        println!("{:<60} {:<70} -> t0 = {}", insn5.code.hex(0, -1, false), insn5.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn6);
        println!("{:<60} {:<70} -> t0 = {}", insn6.code.hex(0, -1, false), insn6.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn7);
        println!("{:<60} {:<70} -> t0 = {}", insn7.code.hex(0, -1, false), insn7.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn8);
        println!("{:<60} {:<70} -> t0 = {}", insn8.code.hex(0, -1, false), insn8.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn9);
        println!("{:<60} {:<70} -> t0 = {}", insn9.code.hex(0, -1, false), insn9.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn10);
        println!("{:<60} {:<70} -> t0 = {}", insn10.code.hex(0, -1, false), insn10.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn11);
        println!("{:<60} {:<70} -> t0 = {}", insn11.code.hex(0, -1, false), insn11.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn12);
        println!("{:<60} {:<70} -> t0 = {}", insn12.code.hex(0, -1, false), insn12.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn13);
        println!("{:<60} {:<70} -> t0 = {}", insn13.code.hex(0, -1, false), insn13.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn14);
        println!("{:<60} {:<70} -> t0 = {}", insn14.code.hex(0, -1, false), insn14.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn15);
        println!("{:<60} {:<70} -> t0 = {}", insn15.code.hex(0, -1, false), insn15.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn16);
        println!("{:<60} {:<70} -> t0 = {}", insn16.code.hex(0, -1, false), insn16.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn17);
        println!("{:<60} {:<70} -> t0 = {}", insn17.code.hex(0, -1, false), insn17.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn18);
        println!("{:<60} {:<70} -> t0 = {}", insn18.code.hex(0, -1, false), insn18.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn19);
        println!("{:<60} {:<70} -> t0 = {}", insn19.code.hex(0, -1, false), insn19.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn20);
        println!("{:<60} {:<70} -> t0 = {}", insn20.code.hex(0, -1, false), insn20.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn21);
        println!("{:<60} {:<70} -> t0 = {}", insn21.code.hex(0, -1, false), insn21.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn22);
        println!("{:<60} {:<70} -> t0 = {}", insn22.code.hex(0, -1, false), insn22.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn23);
        println!("{:<60} {:<70} -> t0 = {}", insn23.code.hex(0, -1, false), insn23.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn24);
        println!("{:<60} {:<70} -> t0 = {}", insn24.code.hex(0, -1, false), insn24.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn25);
        println!("{:<60} {:<70} -> t0 = {}", insn25.code.hex(0, -1, false), insn25.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn26);
        println!("{:<60} {:<70} -> t0 = {}", insn26.code.hex(0, -1, false), insn26.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn27);
        println!("{:<60} {:<70} -> t0 = {}", insn27.code.hex(0, -1, false), insn27.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn28);
        println!("{:<60} {:<70} -> t0 = {}", insn28.code.hex(0, -1, false), insn28.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn29);
        println!("{:<60} {:<70} -> t0 = {}", insn29.code.hex(0, -1, false), insn29.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn30);
        println!("{:<60} {:<70} -> t0 = {}", insn30.code.hex(0, -1, false), insn30.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn31);
        println!("{:<60} {:<70} -> t0 = {}", insn31.code.hex(0, -1, false), insn31.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn32);
        println!("{:<60} {:<70} -> t0 = {}", insn32.code.hex(0, -1, false), insn32.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn33);
        println!("{:<60} {:<70} -> t0 = {}", insn33.code.hex(0, -1, false), insn33.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn34);
        println!("{:<60} {:<70} -> t0 = {}", insn34.code.hex(0, -1, false), insn34.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn35);
        println!("{:<60} {:<70} -> t0 = {}", insn35.code.hex(0, -1, false), insn35.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn36);
        println!("{:<60} {:<70} -> t0 = {}", insn36.code.hex(0, -1, false), insn36.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn37);
        println!("{:<60} {:<70} -> t0 = {}", insn37.code.hex(0, -1, false), insn37.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn38);
        println!("{:<60} {:<70} -> t0 = {}", insn38.code.hex(0, -1, false), insn38.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn39);
        println!("{:<60} {:<70} -> t0 = {}", insn39.code.hex(0, -1, false), insn39.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn40);
        println!("{:<60} {:<70} -> t0 = {}", insn40.code.hex(0, -1, false), insn40.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn41);
        println!("{:<60} {:<70} -> t0 = {}", insn41.code.hex(0, -1, false), insn41.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn42);
        println!("{:<60} {:<70} -> t0 = {}", insn42.code.hex(0, -1, false), insn42.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn43);
        println!("{:<60} {:<70} -> t0 = {}", insn43.code.hex(0, -1, false), insn43.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn44);
        println!("{:<60} {:<70} -> t0 = {}", insn44.code.hex(0, -1, false), insn44.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn45);
        println!("{:<60} {:<70} -> t0 = {}", insn45.code.hex(0, -1, false), insn45.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn46);
        println!("{:<60} {:<70} -> t0 = {}", insn46.code.hex(0, -1, false), insn46.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn47);
        println!("{:<60} {:<70} -> t0 = {}", insn47.code.hex(0, -1, false), insn47.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn48);

        println!("{:<60} {:<70} -> t0 = {}", insn48.code.hex(0, -1, false), insn48.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn49);
        println!("{:<60} {:<70} -> t0 = {}", insn49.code.hex(0, -1, false), insn49.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn50);
        println!("{:<60} {:<70} -> t0 = {}", insn50.code.hex(0, -1, false), insn50.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn51);
        println!("{:<60} {:<70} -> t0 = {}", insn51.code.hex(0, -1, false), insn51.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn52);
        println!("{:<60} {:<70} -> t0 = {}", insn52.code.hex(0, -1, false), insn52.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn53);
        println!("{:<60} {:<70} -> t0 = {}", insn53.code.hex(0, -1, false), insn53.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn54);
        println!("{:<60} {:<70} -> t0 = {}", insn54.code.hex(0, -1, false), insn54.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn55);
        println!("{:<60} {:<70} -> t0 = {}", insn55.code.hex(0, -1, false), insn55.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn56);
        println!("{:<60} {:<70} -> t0 = {}", insn56.code.hex(0, -1, false), insn56.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn57);
        println!("{:<60} {:<70} -> t0 = {}", insn57.code.hex(0, -1, false), insn57.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn58);
        println!("{:<60} {:<70} -> t0 = {}", insn58.code.hex(0, -1, false), insn58.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn59);
        println!("{:<60} {:<70} -> t0 = {}", insn59.code.hex(0, -1, false), insn59.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn60);
        println!("{:<60} {:<70} -> t0 = {}", insn60.code.hex(0, -1, false), insn60.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn61);
        println!("{:<60} {:<70} -> t0 = {}", insn61.code.hex(0, -1, false), insn61.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn62);
        println!("{:<60} {:<70} -> t0 = {}", insn62.code.hex(0, -1, false), insn62.to_string(), cpu.get_nreg("t0").hex(0, -1, false));

        cpu.execute(&insn63);
        println!("{:<60} {:<70} -> t0 = {}", insn63.code.hex(0, -1, false), insn63.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn64);
        println!("{:<60} {:<70} -> t0 = {}", insn64.code.hex(0, -1, false), insn64.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn65);
        println!("{:<60} {:<70} -> t0 = {}", insn65.code.hex(0, -1, false), insn65.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn66);
        println!("{:<60} {:<70} -> t0 = {}", insn66.code.hex(0, -1, false), insn66.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn67);
        println!("{:<60} {:<70} -> t0 = {}", insn67.code.hex(0, -1, false), insn67.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn68);
        println!("{:<60} {:<70} -> t0 = {}", insn68.code.hex(0, -1, false), insn68.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn69);
        println!("{:<60} {:<70} -> t0 = {}", insn69.code.hex(0, -1, false), insn69.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn70);
        println!("{:<60} {:<70} -> t0 = {}", insn70.code.hex(0, -1, false), insn70.to_string(), cpu.get_nreg("t0").hex(0, -1, false));

        cpu.execute(&insn71);
        println!("{:<60} {:<70} -> t0 = {}", insn71.code.hex(0, -1, false), insn71.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn72);
        println!("{:<60} {:<70} -> t0 = {}", insn72.code.hex(0, -1, false), insn72.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn73);
        println!("{:<60} {:<70} -> t0 = {}", insn73.code.hex(0, -1, false), insn73.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn74);
        println!("{:<60} {:<70} -> t0 = {}", insn74.code.hex(0, -1, false), insn74.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn75);
        println!("{:<60} {:<70} -> t0 = {}", insn75.code.hex(0, -1, false), insn75.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn76);
        println!("{:<60} {:<70} -> t0 = {}", insn76.code.hex(0, -1, false), insn76.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn77);
        println!("{:<60} {:<70} -> t0 = {}", insn77.code.hex(0, -1, false), insn77.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn78);
        println!("{:<60} {:<70} -> t0 = {}", insn78.code.hex(0, -1, false), insn78.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn79);
        println!("{:<60} {:<70} -> t0 = {}", insn79.code.hex(0, -1, false), insn79.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn80);
        println!("{:<60} {:<70} -> t0 = {}", insn80.code.hex(0, -1, false), insn80.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn81);
        println!("{:<60} {:<70} -> t0 = {}", insn81.code.hex(0, -1, false), insn81.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn82);
        println!("{:<60} {:<70} -> t0 = {}", insn82.code.hex(0, -1, false), insn82.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn83);
        println!("{:<60} {:<70} -> t0 = {}", insn83.code.hex(0, -1, false), insn83.to_string(), cpu.get_nreg("t0").hex(0, -1, false));
        cpu.execute(&insn84);
        println!("{:<60} {:<70} -> {:?}", insn84.code.hex(0, -1, false), insn84.to_string(), insn84.info());


    }
}