

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT32, BIT64, LITTLE_ENDIAN};
use crate::ir::val::Value;
use crate::ir::insn::{Instruction, OPRS_SIG, OPRS_USD};
use crate::ir::itp::Interpreter;
use crate::ir::mem::CPUThreadStatus;




pub const EVO_ARCH: Arch = Arch::new(ArchKind::EVO, BIT64 | LITTLE_ENDIAN, 128);



/// Insn temp and Reg and Interpreter Pool Init
pub fn evo_itp_init() -> Option<Rc<RefCell<Interpreter>>> {

    // 2. Init regs pool
    Instruction::reg("t0", Value::bit(5, 0));
    Instruction::reg("t1", Value::bit(5, 1));
    Instruction::reg("t2", Value::bit(5, 2));
    Instruction::reg("t3", Value::bit(5, 3));
    Instruction::reg("t4", Value::bit(5, 4));
    Instruction::reg("t5", Value::bit(5, 5));
    Instruction::reg("t6", Value::bit(5, 6));
    Instruction::reg("t7", Value::bit(5, 7));
    Instruction::reg("t8", Value::bit(5, 8));
    Instruction::reg("t9", Value::bit(5, 9));
    Instruction::reg("t10", Value::bit(5, 10));
    Instruction::reg("t11", Value::bit(5, 11));
    Instruction::reg("t12", Value::bit(5, 12));
    Instruction::reg("t13", Value::bit(5, 13));
    Instruction::reg("t14", Value::bit(5, 14));
    Instruction::reg("t15", Value::bit(5, 15));
    Instruction::reg("t16", Value::bit(5, 16));
    Instruction::reg("t17", Value::bit(5, 17));
    Instruction::reg("t18", Value::bit(5, 18));
    Instruction::reg("t19", Value::bit(5, 19));
    Instruction::reg("t20", Value::bit(5, 20));
    Instruction::reg("t21", Value::bit(5, 21));
    Instruction::reg("t22", Value::bit(5, 22));
    Instruction::reg("t23", Value::bit(5, 23));
    Instruction::reg("t24", Value::bit(5, 24));
    Instruction::reg("t25", Value::bit(5, 25));
    Instruction::reg("t26", Value::bit(5, 26));
    Instruction::reg("t27", Value::bit(5, 27));
    Instruction::reg("t28", Value::bit(5, 28));
    Instruction::reg("t29", Value::bit(5, 29));
    Instruction::reg("t30", Value::bit(5, 30));
    Instruction::reg("t31", Value::bit(5, 31));

    // 3. Init insns & insns interpreter
    let itp = Interpreter::def(&EVO_ARCH);
    // EVO Instruction Format:                                                                                                32|31  25|24 20|19 15|  |11  7|6    0|
    // Type: R                                                                                      [rd, rs1, rs2]              |  f7  | rs2 | rs1 |f3|  rd |  op  |
    itp.borrow_mut().def_insn("add_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG,  vec![1, 1, 1], "R", "0B0000000. ........ .000.... .0110011", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Add rs1 and rs2
            let res = rs1.wrapping_add(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("add_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG,  vec![1, 1, 1], "R", "0B0000000. ........ .000.... .0110011", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Add rs1 and rs2
            let res = rs1.wrapping_add(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("sub_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Sub rs1 and rs2
            let res = rs1.wrapping_sub(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("sub_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Sub rs1 and rs2
            let res = rs1.wrapping_sub(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("neg_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = -rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Neg rs1
            let res = rs1.wrapping_neg();
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("neg_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = -rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Neg rs1
            let res = rs1.wrapping_neg();
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("mul_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 * rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Mul rs1 and rs2
            let res = rs1.wrapping_mul(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("mul_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 * rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Mul rs1 and rs2
            let res = rs1.wrapping_mul(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("div_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("div_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("div_u32" , BIT32 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("div_u64" , BIT64 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u64(0);
            // 2. Get rs2(u64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u64(0);
            // 3. Div rs1 and rs2
            let res = rs1.wrapping_div(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("rem_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("rem_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );

    itp.borrow_mut().def_insn("rem_u32" , BIT32 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("rem_u64" , BIT64 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u64(0);
            // 2. Get rs2(u64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u64(0);
            // 3. Rem rs1 and rs2
            let res = rs1.wrapping_rem(rs2);
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("and_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .111.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. And rs1 and rs2
            let res = rs1 & rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("and_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .111.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. And rs1 and rs2
            let res = rs1 & rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("or_i32"  , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .110.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Or rs1 and rs2
            let res = rs1 | rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("or_i64"  , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .110.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Or rs1 and rs2
            let res = rs1 | rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("xor_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Xor rs1 and rs2
            let res = rs1 ^ rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("xor_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Xor rs1 and rs2
            let res = rs1 ^ rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("not_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~rs1 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Not rs1
            let res = !rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("not_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~rs1 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Not rs1
            let res = !rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("andc_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Addc rs1 and rs2
            let res = rs1 & !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("andc_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Addc rs1 and rs2
            let res = rs1 & !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("eqv_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ ~rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Eqv rs1 and rs2
            let res = rs1 ^ !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("eqv_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Eqv rs1 and rs2
            let res = rs1 ^ !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("nand_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 & rs2) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Nand rs1 and rs2
            let res = !(rs1 & rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("nand_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 & rs2) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Nand rs1 and rs2
            let res = !(rs1 & rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("nor_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 | rs2) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Nor rs1 and rs2
            let res = !(rs1 | rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("nor_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 | rs2) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Nor rs1 and rs2
            let res = !(rs1 | rs2);
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("orc_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | ~rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Orc rs1 and rs2
            let res = rs1 | !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("orc_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | ~rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Orc rs1 and rs2
            let res = rs1 | !rs2;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("clz_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? clz(rs1) : rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Clz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.leading_zeros() as i32 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("clz_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? clz(rs1) : rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Clz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.leading_zeros() as i64 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("ctz_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? ctz(rs1) : rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Ctz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.trailing_zeros() as i32 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("ctz_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? ctz(rs1) : rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. Ctz rs1 and rs2
            let res = if rs1 == 0 { rs2 } else { rs1.trailing_zeros() as i64 };
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("shl_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 << rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Shl rs1 and rs2
            let res = rs1 << rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("shl_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 << rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Shl rs1 and rs2
            let res = rs1 << rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("shr_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u32) >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("shr_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u64) >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("sar_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Sar rs1 and rs2
            let res = rs1 >> rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("sar_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Sar rs1 and rs2
            let res = rs1 >> rs2;
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("rol_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Rotl rs1 and rs2
            let res = rs1.rotate_left(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("rol_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Rotl rs1 and rs2
            let res = rs1.rotate_left(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("ror_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. rs2 < 0 || rs2 >= 32 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 32 { 0 } else { rs2 };
            // 4. Rotr rs1 and rs2
            let res = rs1.rotate_right(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("ror_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get rs2(i64)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i64(0);
            // 3. rs2 < 0 || rs2 >= 64 -> 0
            let rs2 = if rs2 < 0 || rs2 >= 64 { 0 } else { rs2 };
            // 4. Rotr rs1 and rs2
            let res = rs1.rotate_right(rs2 as u32);
            // 5. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("mov_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Mov rs1
            let res = rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("mov_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Mov rs1
            let res = rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extb_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i8 -> i32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i8(0);
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("extb_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i8 -> i64) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i8(0);
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extb_u32" , BIT32 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u8 -> u32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u8(0);
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("extb_u64" , BIT64 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u8 -> u64) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u8(0);
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("exth_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i16 -> i32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i16(0);
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("exth_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i16 -> i64) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i16(0);
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("exth_u32" , BIT32 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u16 -> u32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("exth_u64" , BIT64 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u16 -> u64) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("extw_i32" , BIT32 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> i32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("extw_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> i64) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extw_u32" , BIT32 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u32 -> u32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("extw_u64" , BIT64 | LITTLE_ENDIAN | OPRS_USD, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u32 -> u64) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("extdl_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> low i32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Extract double low 32-bit
            let res = ((rs1 << 32) >> 32) as u32;
            // 3. Set rd(i64[31:0])
            let mut rd = proc0.get_reg(insn.rd() as usize);
            rd.set_uword(0, 0, 32, res);
            proc0.set_reg(insn.rd() as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("extdh_i64" , BIT64 | LITTLE_ENDIAN | OPRS_SIG, vec![1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> low i32) ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Extract double high 32-bit
            let res = (rs1 >> 32) as u32;
            // 3. Set rd(i64[31:0])
            let mut rd = proc0.get_reg(insn.rd() as usize);
            rd.set_uword(0, 0, 32, res);
            proc0.set_reg(insn.rd() as usize, rd);
        }
    );



    itp.borrow_mut().def_insn("slt" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .010.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 < rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Slt rs1 and rs2
            let res = if rs1 < rs2 { 1 } else { 0 };
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("sltu", BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .011.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 < rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Sltu rs1 and rs2
            let res = if rs1 < rs2 { 1 } else { 0 };
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    // Type: I                                [rd, rs1, imm]                 |    imm     | rs1 |f3|  rd |  op  |
    itp.borrow_mut().def_insn("addi", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .000.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 + imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Add rs1 and imm
            let res = rs1 + imm;
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("xori", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .100.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 ^ imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Xor rs1 and imm
            let res = rs1 ^ imm;
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("ori" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .110.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 | imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Or rs1 and imm
            let res = rs1 | imm;
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("andi", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .111.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 & imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. And rs1 and imm
            let res = rs1 & imm;
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("slli", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B0000000. ........ .001.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 << imm[0:4] ======= //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get imm[0:4](u32)
            let imm = (insn.imm_i() & 0x1F) as u32;
            // 3. Sll rs1 and imm
            let res = rs1 << imm;
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("srli", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B0000000. ........ .101.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 >> imm[0:4] ======= //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get imm[0:4](u32)
            let imm = (insn.imm_i() & 0x1F) as u32;
            // 3. Sll rs1 and imm
            let res = rs1 >> imm;
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("srai", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B0100000. ........ .101.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 >> imm[0:4] ======= //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get imm[0:4](u32)
            let imm = (insn.imm_i() & 0x1F) as u32;
            // 3. Sll rs1 and imm
            let res = rs1 >> imm;
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("slti", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .010.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 < imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Slt rs1 and imm
            let res = if rs1 < imm { 1 } else { 0 };
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("sltiu", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .011.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 < imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get imm(u32)
            let imm = insn.imm_i() as u32;
            // 3. Slt rs1 and imm
            let res = if rs1 < imm { 1 } else { 0 };
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("lb", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Byte
            let val = proc0.read_mem((rs1 + imm) as usize, 1).get_byte(0);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
        }
    );
    itp.borrow_mut().def_insn("lh", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .001.... .0000011", 
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Half
            let val = proc0.read_mem((rs1 + imm) as usize, 1).get_half(0);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
        }
    );
    itp.borrow_mut().def_insn("lw", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .010.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Word
            let val = proc0.read_mem((rs1 + imm) as usize, 1).get_word(0);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
        }
    );
    itp.borrow_mut().def_insn("lbu", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .100.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Byte
            let val = proc0.read_mem((rs1 + imm) as usize, 1).get_byte(0);
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(val as u32));
        }
    );
    itp.borrow_mut().def_insn("lhu", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .101.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Half
            let val = proc0.read_mem((rs1 + imm) as usize, 1).get_half(0);
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(val as u32));
        }
    );
    itp.borrow_mut().def_insn("jalr", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "I", "0B........ ........ .000.... .1100111",
        |cpu, insn| {
            // ======== rd = pc + 4; pc = rs1 + imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(proc0.get_pc().get_i32(0) + 4));
            // 4. Set pc(i32)
            proc0.set_pc(Value::i32(rs1 + imm));
        }
    );
    itp.borrow_mut().def_insn("ecall", BIT32 | LITTLE_ENDIAN, vec![], "I", "0B00000000 0000.... .000.... .1110111",
        |cpu, insn| {
            // ======== ecall ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // System will do next part according to register `a7`(t17)
            proc0.set_status(CPUThreadStatus::Blocked);
        }
    );
    itp.borrow_mut().def_insn("ebreak", BIT32 | LITTLE_ENDIAN, vec![], "I", "0B00000000 0001.... .000.... .1110111",
        |cpu, insn| {
            // ======== ebreak ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // Debugger will do next part according to register `a7`(t17)
            proc0.set_status(CPUThreadStatus::Blocked);
        }
    );
    // Type:S 
    itp.borrow_mut().def_insn("sb", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "S", "0B........ ........ .000.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_byte(0) as i8;
            // 4. Write Mem: Byte
            proc0.write_mem((rs1 + imm) as usize, Value::i8(rs2));
        }
    );
    itp.borrow_mut().def_insn("sh", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "S", "0B........ ........ .001.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_half(0) as i16;
            // 4. Write Mem: Half
            proc0.write_mem((rs1 + imm) as usize, Value::i16(rs2));
        }
    );
    itp.borrow_mut().def_insn("sw", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "S", "0B........ ........ .010.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_word(0) as i32;
            // 4. Write Mem: Word
            proc0.write_mem((rs1 + imm) as usize, Value::i32(rs2));
        }
    );
    // Type: B
    itp.borrow_mut().def_insn("beq", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "B", "0B........ ........ .000.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 == rs2) pc += imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Get imm(i32)
            let imm = insn.imm_b() as i32;
            // 4. Set PC
            if rs1 == rs2 {
                proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
            }
        }
    );
    itp.borrow_mut().def_insn("bne", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "B", "0B........ ........ .001.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 != rs2) pc += imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Get imm(i32)
            let imm = insn.imm_b() as i32;
            // 4. Set PC
            if rs1 != rs2 {
                proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
            }
        }
    );
    itp.borrow_mut().def_insn("blt", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "B", "0B........ ........ .100.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 < rs2) pc += imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Get imm(i32)
            let imm = insn.imm_b() as i32;
            // 4. Set PC
            if rs1 < rs2 {
                proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
            }
        }
    );
    itp.borrow_mut().def_insn("bge", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "B", "0B........ ........ .101.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 >= rs2) pc += imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Get imm(i32)
            let imm = insn.imm_b() as i32;
            // 4. Set PC
            if rs1 >= rs2 {
                proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
            }
        }
    );
    itp.borrow_mut().def_insn("bltu", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "B", "0B........ ........ .110.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 < rs2) pc += imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Get imm(i32)
            let imm = insn.imm_b() as i32;
            // 4. Set PC
            if rs1 < rs2 {
                proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
            }
        }
    );
    itp.borrow_mut().def_insn("bgeu", BIT32 | LITTLE_ENDIAN, vec![1, 1, 0], "B", "0B........ ........ .111.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 >= rs2) pc += imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Get imm(i32)
            let imm = insn.imm_b() as i32;
            // 4. Set PC
            if rs1 >= rs2 {
                proc0.set_pc(Value::i32(proc0.get_pc().get_i32(0) + imm));
            }
        }
    );
    // Type: U
    itp.borrow_mut().def_insn("lui", BIT32 | LITTLE_ENDIAN, vec![1, 0], "U", "0B........ ........ ........ .0110111",
        |cpu, insn| {
            // ======== rd = imm << 12 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get imm(i32)
            let imm = insn.imm_u() as i32;
            // 2. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(imm << 12));
        }
    );
    itp.borrow_mut().def_insn("auipc", BIT32 | LITTLE_ENDIAN, vec![1, 0], "U", "0B........ ........ ........ .0010111",
        |cpu, insn| {
            // ======== rd = pc + imm << 12 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get imm(i32)
            let imm = insn.imm_u() as i32;
            // 2. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(proc0.get_pc().get_i32(0) + imm << 12));
        }
    );
    // Type: J
    itp.borrow_mut().def_insn("jal", BIT32 | LITTLE_ENDIAN, vec![1, 0], "J", "0B........ ........ ........ .1101111",
        |cpu, insn| {
            // ======== rd = pc + 4; pc = pc + imm ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get imm(i32)
            let imm = insn.imm_j() as i32;
            // 2. Get pc(i32)
            let pc = proc0.get_pc().get_i32(0);
            // 3. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(pc + 4));
            // 4. Set PC
            proc0.set_pc(Value::i32(pc + imm));
        }
    );


    Some(itp)
}




#[cfg(test)]
mod evo_test {

    use super::*;
    use crate::ir::cpu::CPUState;


    #[test]
    fn evo_itp() {
        let cpu = CPUState::init(&EVO_ARCH, &EVO_ARCH, None, None, None);
        cpu.set_nreg("t1", Value::i64(23));
        cpu.set_nreg("t2", Value::i64(17));
        cpu.set_nreg("t3", Value::i64(65535));
        cpu.write_mem(26, Value::i32(0x1ffff));

        // R-Type Insns Test
        let insn1  = Instruction::from_string("add_i32 t0, t1, t2");
        let insn2  = Instruction::from_string("add_i64 t0, t1, t2");
        let insn3  = Instruction::from_string("sub_i32 t0, t1, t2");
        let insn4  = Instruction::from_string("sub_i64 t0, t1, t2");
        let insn5  = Instruction::from_string("neg_i32 t0, t1");
        let insn6  = Instruction::from_string("neg_i64 t0, t1");
        let insn7  = Instruction::from_string("mul_i32 t0, t1, t2");
        let insn8  = Instruction::from_string("mul_i64 t0, t1, t2");
        let insn9  = Instruction::from_string("div_i32 t0, t1, t2");
        let insn10 = Instruction::from_string("div_i64 t0, t1, t2");
        let insn11 = Instruction::from_string("div_u32 t0, t1, t2");
        let insn12 = Instruction::from_string("div_u64 t0, t1, t2");
        let insn13 = Instruction::from_string("rem_i32 t0, t1, t2");
        let insn14 = Instruction::from_string("rem_i64 t0, t1, t2");
        let insn15 = Instruction::from_string("rem_u32 t0, t1, t2");
        let insn16 = Instruction::from_string("rem_u64 t0, t1, t2");

        let insn17 = Instruction::from_string("and_i32 t0, t1, t2");
        let insn18 = Instruction::from_string("and_i64 t0, t1, t2");
        let insn19 = Instruction::from_string("or_i32 t0, t1, t2");
        let insn20 = Instruction::from_string("or_i64 t0, t1, t2");
        let insn21 = Instruction::from_string("xor_i32 t0, t1, t2");
        let insn22 = Instruction::from_string("xor_i64 t0, t1, t2");
        let insn23 = Instruction::from_string("not_i32 t0, t1");
        let insn24 = Instruction::from_string("not_i64 t0, t1");
        let insn25 = Instruction::from_string("andc_i32 t0, t1, t2");
        let insn26 = Instruction::from_string("andc_i64 t0, t1, t2");
        let insn27 = Instruction::from_string("eqv_i32 t0, t1, t2");
        let insn28 = Instruction::from_string("eqv_i64 t0, t1, t2");
        let insn29 = Instruction::from_string("nand_i32 t0, t1, t2");
        let insn30 = Instruction::from_string("nand_i64 t0, t1, t2");
        let insn31 = Instruction::from_string("nor_i32 t0, t1, t2");
        let insn32 = Instruction::from_string("nor_i64 t0, t1, t2");
        let insn33 = Instruction::from_string("orc_i32 t0, t1, t2");
        let insn34 = Instruction::from_string("orc_i64 t0, t1, t2");

        let insn35 = Instruction::from_string("clz_i32 t0, t1, t2");
        let insn36 = Instruction::from_string("clz_i64 t0, t1, t2");
        let insn37 = Instruction::from_string("ctz_i32 t0, t1, t2");
        let insn38 = Instruction::from_string("ctz_i64 t0, t1, t2");
        let insn39 = Instruction::from_string("shl_i32 t0, t1, t2");
        let insn40 = Instruction::from_string("shl_i64 t0, t1, t2");
        let insn41 = Instruction::from_string("shr_i32 t0, t1, t2");
        let insn42 = Instruction::from_string("shr_i64 t0, t1, t2");
        let insn43 = Instruction::from_string("sar_i32 t0, t1, t2");
        let insn44 = Instruction::from_string("sar_i64 t0, t1, t2");
        let insn45 = Instruction::from_string("rol_i32 t0, t1, t2");
        let insn46 = Instruction::from_string("rol_i64 t0, t1, t2");
        let insn47 = Instruction::from_string("ror_i32 t0, t1, t2");
        let insn48 = Instruction::from_string("ror_i64 t0, t1, t2");
        let insn49 = Instruction::from_string("mov_i32 t0, t1, t2");
        let insn50 = Instruction::from_string("mov_i64 t0, t1, t2");

        let insn51 = Instruction::from_string("extb_i32 t0, t3");
        let insn52 = Instruction::from_string("extb_i64 t0, t3");
        let insn53 = Instruction::from_string("extb_u32 t0, t3");
        let insn54 = Instruction::from_string("extb_u64 t0, t3");
        let insn55 = Instruction::from_string("exth_i32 t0, t3");
        let insn56 = Instruction::from_string("exth_i64 t0, t3");
        let insn57 = Instruction::from_string("exth_u32 t0, t3");
        let insn58 = Instruction::from_string("exth_u64 t0, t3");
        let insn59 = Instruction::from_string("extw_i32 t0, t3");
        let insn60 = Instruction::from_string("extw_i64 t0, t3");
        let insn61 = Instruction::from_string("extw_u32 t0, t3");
        let insn62 = Instruction::from_string("extw_u64 t0, t3");

        let insn63 = Instruction::from_string("extdh_i64 t0, t1");
        let insn64 = Instruction::from_string("extdl_i64 t0, t1");

        cpu.execute(&insn1);
        println!("{:<50} -> t0 = {}", insn1.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn2);
        println!("{:<50} -> t0 = {}", insn2.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn3);
        println!("{:<50} -> t0 = {}", insn3.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn4);
        println!("{:<50} -> t0 = {}", insn4.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn5);
        println!("{:<50} -> t0 = {}", insn5.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn6);
        println!("{:<50} -> t0 = {}", insn6.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn7);
        println!("{:<50} -> t0 = {}", insn7.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn8);
        println!("{:<50} -> t0 = {}", insn8.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn9);
        println!("{:<50} -> t0 = {}", insn9.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn10);
        println!("{:<50} -> t0 = {}", insn10.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn11);
        println!("{:<50} -> t0 = {}", insn11.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn12);
        println!("{:<50} -> t0 = {}", insn12.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn13);
        println!("{:<50} -> t0 = {}", insn13.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn14);
        println!("{:<50} -> t0 = {}", insn14.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn15);
        println!("{:<50} -> t0 = {}", insn15.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn16);

        println!("{:<50} -> t0 = {}", insn16.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn17);
        println!("{:<50} -> t0 = {}", insn17.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn18);
        println!("{:<50} -> t0 = {}", insn18.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn19);
        println!("{:<50} -> t0 = {}", insn19.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn20);
        println!("{:<50} -> t0 = {}", insn20.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn21);
        println!("{:<50} -> t0 = {}", insn21.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn22);
        println!("{:<50} -> t0 = {}", insn22.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn23);
        println!("{:<50} -> t0 = {}", insn23.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn24);
        println!("{:<50} -> t0 = {}", insn24.to_string(), cpu.get_nreg("t0").get_i64(0));

        cpu.execute(&insn25);
        println!("{:<50} -> t0 = {}", insn25.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn26);
        println!("{:<50} -> t0 = {}", insn26.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn27);
        println!("{:<50} -> t0 = {}", insn27.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn28);
        println!("{:<50} -> t0 = {}", insn28.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn29);
        println!("{:<50} -> t0 = {}", insn29.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn30);
        println!("{:<50} -> t0 = {}", insn30.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn31);
        println!("{:<50} -> t0 = {}", insn31.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn32);
        println!("{:<50} -> t0 = {}", insn32.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn33);
        println!("{:<50} -> t0 = {}", insn33.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn34);
        println!("{:<50} -> t0 = {}", insn34.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn35);
        println!("{:<50} -> t0 = {}", insn35.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn36);
        println!("{:<50} -> t0 = {}", insn36.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn37);
        println!("{:<50} -> t0 = {}", insn37.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn38);
        println!("{:<50} -> t0 = {}", insn38.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn39);
        println!("{:<50} -> t0 = {}", insn39.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn40);
        println!("{:<50} -> t0 = {}", insn40.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn41);
        println!("{:<50} -> t0 = {}", insn41.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn42);
        println!("{:<50} -> t0 = {}", insn42.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn43);
        println!("{:<50} -> t0 = {}", insn43.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn44);
        println!("{:<50} -> t0 = {}", insn44.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn45);
        println!("{:<50} -> t0 = {}", insn45.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn46);
        println!("{:<50} -> t0 = {}", insn46.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn47);
        println!("{:<50} -> t0 = {}", insn47.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn48);
        println!("{:<50} -> t0 = {}", insn48.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn49);
        println!("{:<50} -> t0 = {}", insn49.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn50);
        println!("{:<50} -> t0 = {}", insn50.to_string(), cpu.get_nreg("t0").get_i64(0));

        cpu.execute(&insn51);
        println!("{:<50} -> t0 = {}", insn51.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn52);
        println!("{:<50} -> t0 = {}", insn52.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn53);
        println!("{:<50} -> t0 = {}", insn53.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn54);
        println!("{:<50} -> t0 = {}", insn54.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn55);
        println!("{:<50} -> t0 = {}", insn55.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn56);
        println!("{:<50} -> t0 = {}", insn56.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn57);
        println!("{:<50} -> t0 = {}", insn57.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn58);
        println!("{:<50} -> t0 = {}", insn58.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn59);
        println!("{:<50} -> t0 = {}", insn59.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn60);
        println!("{:<50} -> t0 = {}", insn60.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn61);
        println!("{:<50} -> t0 = {}", insn61.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn62);
        println!("{:<50} -> t0 = {}", insn62.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn63);
        println!("{:<50} -> t0 = {}", insn63.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn64);
        println!("{:<50} -> t0 = {}", insn64.to_string(), cpu.get_nreg("t0").get_i64(0));


        // cpu.set_nreg("t0", Value::i32(56));
        // cpu.execute(&insn22);
        // println!("{:<50} -> mem = {}", insn22.to_string(), cpu.read_mem(26, 1).bin(0, 1, false));
        // cpu.set_nreg("t0", Value::i32(732));
        // cpu.execute(&insn23);
        // println!("{:<50} -> mem = {}", insn23.to_string(), cpu.read_mem(26, 1).bin(0, 2, false));
        // cpu.set_nreg("t0", Value::i32(-8739));
        // cpu.execute(&insn24);
        // println!("{:<50} -> mem = {}", insn24.to_string(), cpu.read_mem(26, 1).bin(0, 4, false));

        // cpu.execute(&insn31);
        // println!("{:<50} -> pc = {}, t0 = {}", insn31.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));
        // cpu.execute(&insn32);
        // println!("{:<50} -> pc = {}, t0 = {}", insn32.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));

        // cpu.execute(&insn33);
        // println!("{:<50} -> pc = {}, t0 = {}", insn33.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));
        // cpu.execute(&insn34);
        // println!("{:<50} -> pc = {}, t0 = {}", insn34.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));

        // cpu.execute(&insn35);
        // println!("{:<50} -> status = {}", insn35.to_string(), cpu.status());
        // cpu.execute(&insn36);
        // println!("{:<50} -> status = {}", insn36.to_string(), cpu.status());
    }
}