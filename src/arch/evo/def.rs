

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::{log_error, log_info, log_warning};
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT32, BIT64, LITTLE_ENDIAN};
use crate::core::val::Value;
use crate::core::op::{OpcodeKind, Operand, OPR_IMM, OPR_REG};
use crate::core::insn::{Instruction, RegFile, INSN_SIG, INSN_USD};
use crate::core::itp::Interpreter;
use crate::core::mem::CPUThreadStatus;



// ============================================================================== //
//                             evo::def::arch
// ============================================================================== //

pub const EVO_ARCH: Arch = Arch::new(ArchKind::EVO, BIT64 | LITTLE_ENDIAN, 258);




// ============================================================================== //
//                          evo::def::interpreter
// ============================================================================== //

/// Insn temp and Reg and Interpreter Pool Init
pub fn evo_itp_init() -> Option<Rc<RefCell<Interpreter>>> {

    // 1. Init regs pool
    RegFile::def(&EVO_ARCH, "t0", Value::bit(5, 0), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t1", Value::bit(5, 1), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t2", Value::bit(5, 2), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t3", Value::bit(5, 3), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t4", Value::bit(5, 4), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t5", Value::bit(5, 5), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t6", Value::bit(5, 6), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t7", Value::bit(5, 7), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t8", Value::bit(5, 8), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t9", Value::bit(5, 9), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t10", Value::bit(5, 10), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t11", Value::bit(5, 11), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t12", Value::bit(5, 12), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t13", Value::bit(5, 13), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t14", Value::bit(5, 14), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t15", Value::bit(5, 15), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t16", Value::bit(5, 16), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t17", Value::bit(5, 17), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t18", Value::bit(5, 18), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t19", Value::bit(5, 19), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t20", Value::bit(5, 20), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t21", Value::bit(5, 21), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t22", Value::bit(5, 22), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t23", Value::bit(5, 23), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t24", Value::bit(5, 24), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t25", Value::bit(5, 25), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t26", Value::bit(5, 26), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t27", Value::bit(5, 27), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t28", Value::bit(5, 28), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t29", Value::bit(5, 29), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t30", Value::bit(5, 30), BIT64 | LITTLE_ENDIAN);
    RegFile::def(&EVO_ARCH, "t31", Value::bit(5, 31), BIT64 | LITTLE_ENDIAN);

    // 2. Init insns & insns interpreter
    let itp = Interpreter::def(&EVO_ARCH);
    // EVO Instruction Format:                                                                                                32|31  25|24 20|19 15|  |11  7|6    0|
    // Type: R                                                                                      [rd, rs1, rs2]              |  f7  | rs2 | rs1 |f3|  rd |  op  |
    itp.borrow_mut().def_insn("add_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG,  vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .000.... .0110011", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //

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
    itp.borrow_mut().def_insn("add_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG,  vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .000.... .0110011", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //
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
    itp.borrow_mut().def_insn("sub_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
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
    itp.borrow_mut().def_insn("sub_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
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
    itp.borrow_mut().def_insn("neg_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = -rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Neg rs1
            let res = rs1.wrapping_neg();
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("neg_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = -rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Neg rs1
            let res = rs1.wrapping_neg();
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("mul_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 * rs2 ======== //
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
    itp.borrow_mut().def_insn("mul_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 * rs2 ======== //
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
    itp.borrow_mut().def_insn("div_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
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
    itp.borrow_mut().def_insn("div_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
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
    itp.borrow_mut().def_insn("div_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
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
    itp.borrow_mut().def_insn("div_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 / rs2 ======== //
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
    itp.borrow_mut().def_insn("rem_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
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
    itp.borrow_mut().def_insn("rem_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
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
    itp.borrow_mut().def_insn("rem_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
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
    itp.borrow_mut().def_insn("rem_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 % rs2 ======== //
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
    itp.borrow_mut().def_insn("and_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .111.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
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
    itp.borrow_mut().def_insn("and_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .111.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
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
    itp.borrow_mut().def_insn("or_i32"  , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .110.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
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
    itp.borrow_mut().def_insn("or_i64"  , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .110.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
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
    itp.borrow_mut().def_insn("xor_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
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
    itp.borrow_mut().def_insn("xor_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
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
    itp.borrow_mut().def_insn("not_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Not rs1
            let res = !rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("not_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Not rs1
            let res = !rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("andc_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
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
    itp.borrow_mut().def_insn("andc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
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
    itp.borrow_mut().def_insn("eqv_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ ~rs2 ======== //
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
    itp.borrow_mut().def_insn("eqv_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & ~rs2 ======== //
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
    itp.borrow_mut().def_insn("nand_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 & rs2) ======== //
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
    itp.borrow_mut().def_insn("nand_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 & rs2) ======== //
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
    itp.borrow_mut().def_insn("nor_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 | rs2) ======== //
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
    itp.borrow_mut().def_insn("nor_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = ~(rs1 | rs2) ======== //
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
    itp.borrow_mut().def_insn("orc_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | ~rs2 ======== //
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
    itp.borrow_mut().def_insn("orc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | ~rs2 ======== //
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
    itp.borrow_mut().def_insn("clz_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? clz(rs1) : rs2 ======== //
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
    itp.borrow_mut().def_insn("clz_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? clz(rs1) : rs2 ======== //
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
    itp.borrow_mut().def_insn("ctz_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? ctz(rs1) : rs2 ======== //
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
    itp.borrow_mut().def_insn("ctz_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ? ctz(rs1) : rs2 ======== //
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
    itp.borrow_mut().def_insn("shl_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 << rs2 ======== //
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
    itp.borrow_mut().def_insn("shl_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 << rs2 ======== //
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
    itp.borrow_mut().def_insn("shr_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
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
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("shr_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
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
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("sar_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
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
    itp.borrow_mut().def_insn("sar_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
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
    itp.borrow_mut().def_insn("rol_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
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
    itp.borrow_mut().def_insn("rol_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
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
    itp.borrow_mut().def_insn("ror_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
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
    itp.borrow_mut().def_insn("ror_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 >> rs2 ======== //
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
    itp.borrow_mut().def_insn("mov_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Mov rs1
            let res = rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("mov_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Mov rs1
            let res = rs1;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extb_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i8 -> i32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i8(0);
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("extb_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i8 -> i64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i8(0);
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extb_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u8 -> u32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u8(0);
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("extb_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u8 -> u64) ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u8)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u8(0);
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("exth_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i16 -> i32) ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i16(0);
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("exth_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i16 -> i64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i16(0);
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("exth_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u16 -> u32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("exth_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u16 -> u64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u16)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("extw_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> i32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Ext rs1 to i32
            let res = rs1 as i32;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res as i64));
        }
    );
    itp.borrow_mut().def_insn("extw_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> i64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Ext rs1 to i64
            let res = rs1 as i64;
            // 3. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::i64(res));
        }
    );
    itp.borrow_mut().def_insn("extw_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u32 -> u32) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Ext rs1 to u32
            let res = rs1 as u32;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("extw_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(u32 -> u64) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Ext rs1 to u64
            let res = rs1 as u64;
            // 3. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("etr_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr_i32 rd, rs1, pos, len");
        }
    );
    itp.borrow_mut().def_insn("etr_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr_i64 rd, rs1, pos, len");
        }
    );
    itp.borrow_mut().def_insn("etr_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr_u32 rd, rs1, pos, len");
        }
    );
    itp.borrow_mut().def_insn("etr_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr_u64 rd, rs1, pos, len");
        }
    );

    itp.borrow_mut().def_insn("etrl_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> low half) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Set rd(i64[15:0])
            let mut rd = proc0.get_reg(insn.rd() as usize);
            rd.set_uhalf(0, 0, 16, rs1);
            proc0.set_reg(insn.rd() as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("etrh_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> high half) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1 high 16-bit word
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(1);
            // 2. Set rd(i64[15:0])
            let mut rd = proc0.get_reg(insn.rd() as usize);
            rd.set_uhalf(0, 0, 16, rs1);
            proc0.set_reg(insn.rd() as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("etrl_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> low word) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1 low 32-bit word
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Set rd(i64[31:0])
            let mut rd = proc0.get_reg(insn.rd() as usize);
            rd.set_uword(0, 0, 32, rs1);
            proc0.set_reg(insn.rd() as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("etrh_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> high word) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1 high 32-bit word
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(4);
            // 2. Set rd(i64[31:0])
            let mut rd = proc0.get_reg(insn.rd() as usize);
            rd.set_uword(0, 0, 32, rs1);
            proc0.set_reg(insn.rd() as usize, rd);
        }
    );
    itp.borrow_mut().def_insn("truc_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i32 -> low half + fill 0) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i16(0);
            // 2. set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(rs1 as u64));
        }
    );
    itp.borrow_mut().def_insn("truc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(i64 -> low word + fill 0) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1 low 32-bit word
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(rs1 as u64));
        }
    );
    itp.borrow_mut().def_insn("conc_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = (rs2 << 32) | rs1 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1 32-bit word as low
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0) as u64;
            // 2. Get rs2 32-bit word as high
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0) as u64;
            // 3. Concat rs1 low 32-bit word and rs2 high 32-bit word
            let res = (rs2 << 32) | rs1;
            // 3. set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("conn_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1(low word) | rs2(high word) ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1 low 32-bit word
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0) as u64;
            // 2. Get rs2 high 32-bit word
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(4) as u64;
            // 3. Concat rs1 low 32-bit word and rs2 high 32-bit word
            let res = (rs2 << 32) | rs1;
            // 3. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("swph_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Reverse byte
            let res = u16::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swph_i64" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u16(0);
            // 2. Reverse byte
            let res = u16::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swpw_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Reverse byte
            let res = u32::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swpw_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Reverse byte
            let res = u32::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res as u64));
        }
    );
    itp.borrow_mut().def_insn("swpd_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1.byte.reverse ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u64(0);
            // 2. Reverse byte
            let res = u64::from_le_bytes(rs1.to_be_bytes());
            // 3. Set rd
            proc0.set_reg(insn.rd() as usize, Value::u64(res));
        }
    );
    itp.borrow_mut().def_insn("depo_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd, rs1, rs2, ofs, len ======== //
            log_info!("TODO: depo_i32 rd, rs1, rs2, ofs, len");
        }
    );
    itp.borrow_mut().def_insn("depo_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd, rs1, rs2, ofs, len ======== //
            log_info!("TODO: depo_i64 rd, rs1, rs2, ofs, len");
        }
    );
    itp.borrow_mut().def_insn("ldb_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_byte(0) as i32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldb_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_byte(0) as i64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldb_u32", BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_byte(0) as u32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldb_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_byte(0) as u64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val));
        }
    );
    itp.borrow_mut().def_insn("ldh_i32", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_half(0) as i32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldh_i64", BIT64 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_half(0) as i64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldh_u32", BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_half(0) as u32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldh_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_half(0) as u64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val));
        }
    );
    itp.borrow_mut().def_insn("ldw_i32", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_word(0) as i32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldw_i64", BIT64 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_word(0) as i64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldw_u32", BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_word(0) as u32;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldw_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_word(0) as u64;
            // 4. Set rd(u64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val));
        }
    );
    itp.borrow_mut().def_insn("ldd_i64", BIT64 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Dword
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_dword(0) as i64;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val as u64));
        }
    );
    itp.borrow_mut().def_insn("ldd_u64", BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_i() as i64;
            // 3. Read Mem: Dword
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_dword(0) as u64;
            // 4. Set rd(i64)
            proc0.set_reg(insn.rd() as usize, Value::u64(val));
        }
    );
    // Type:S 
    itp.borrow_mut().def_insn("stb_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .000.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_byte(0) as i8;
            // 4. Write Mem: Byte
            proc0.mem_write((rs1 + imm) as usize, Value::i8(rs2));
        }
    );
    itp.borrow_mut().def_insn("stb_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .000.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_s() as i64;
            // 3. Get rs2
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_byte(0) as i8;
            // 4. Write Mem: Byte
            proc0.mem_write((rs1 + imm) as usize, Value::i8(rs2));
        }
    );
    itp.borrow_mut().def_insn("sth_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .001.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_half(0) as i16;
            // 4. Write Mem: Half
            proc0.mem_write((rs1 + imm) as usize, Value::i16(rs2));
        }
    );
    itp.borrow_mut().def_insn("sth_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .001.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_s() as i64;
            // 3. Get rs2
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_half(0) as i16;
            // 4. Write Mem: Half
            proc0.mem_write((rs1 + imm) as usize, Value::i16(rs2));
        }
    );
    itp.borrow_mut().def_insn("stw_i32", BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .010.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_word(0) as i32;
            // 4. Write Mem: Word
            proc0.mem_write((rs1 + imm) as usize, Value::i32(rs2));
        }
    );
    itp.borrow_mut().def_insn("stw_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .010.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_s() as i64;
            // 3. Get rs2
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_word(0) as i32;
            // 4. Write Mem: Word
            proc0.mem_write((rs1 + imm) as usize, Value::i32(rs2));
        }
    );
    itp.borrow_mut().def_insn("std_i64", BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .010.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i64)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i64(0);
            // 2. Get imm(i64)
            let imm = insn.imm_s() as i64;
            // 3. Get rs2
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_dword(0) as i64;
            // 4. Write Mem: DWord
            proc0.mem_write((rs1 + imm) as usize, Value::i64(rs2));
        }
    );

    itp.borrow_mut().def_insn("slt" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .010.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 < rs2 ======== //
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
    itp.borrow_mut().def_insn("sltu", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .011.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 < rs2 ======== //
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
    itp.borrow_mut().def_insn("addi", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 + imm ======== //
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
    itp.borrow_mut().def_insn("xori", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .100.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 ^ imm ======== //
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
    itp.borrow_mut().def_insn("ori" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .110.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 | imm ======== //
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
    itp.borrow_mut().def_insn("andi", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .111.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 & imm ======== //
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
    itp.borrow_mut().def_insn("slli", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B0000000. ........ .001.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 << imm[0:4] ======= //
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
    itp.borrow_mut().def_insn("srli", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B0000000. ........ .101.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 >> imm[0:4] ======= //
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
    itp.borrow_mut().def_insn("srai", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B0100000. ........ .101.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 >> imm[0:4] ======= //
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
    itp.borrow_mut().def_insn("slti", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .010.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 < imm ======== //
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
    itp.borrow_mut().def_insn("sltiu", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .011.... .0010011",
        |cpu, insn| {
            // ======== rd = rs1 < imm ======== //
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

    itp.borrow_mut().def_insn("jalr", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .1100111",
        |cpu, insn| {
            // ======== rd = pc + 4; pc = rs1 + imm ======== //
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
        |cpu, _| {
            // ======== ecall ======== //
            let proc0 = cpu.proc.borrow().clone();
            // System will do next part according to register `a7`(t17)
            proc0.set_status(CPUThreadStatus::Blocked);
        }
    );
    itp.borrow_mut().def_insn("ebreak", BIT32 | LITTLE_ENDIAN, vec![], "I", "0B00000000 0001.... .000.... .1110111",
        |cpu, _| {
            // ======== ebreak ======== //
            let proc0 = cpu.proc.borrow().clone();
            // Debugger will do next part according to register `a7`(t17)
            proc0.set_status(CPUThreadStatus::Blocked);
        }
    );

    // Type: B
    itp.borrow_mut().def_insn("beq", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "B", "0B........ ........ .000.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 == rs2) pc += imm ======== //
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
    itp.borrow_mut().def_insn("bne", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "B", "0B........ ........ .001.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 != rs2) pc += imm ======== //
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
    itp.borrow_mut().def_insn("blt", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "B", "0B........ ........ .100.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 < rs2) pc += imm ======== //
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
    itp.borrow_mut().def_insn("bge", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "B", "0B........ ........ .101.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 >= rs2) pc += imm ======== //
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
    itp.borrow_mut().def_insn("bltu", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "B", "0B........ ........ .110.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 < rs2) pc += imm ======== //
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
    itp.borrow_mut().def_insn("bgeu", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "B", "0B........ ........ .111.... .1100011",
        |cpu, insn| {
            // ======== if(rs1 >= rs2) pc += imm ======== //
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
    
    itp.borrow_mut().def_insn("add2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] + [rs2_high: rs2_low] ======== //
            log_info!("TODO: add2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("add2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] + [rs2_high: rs2_low] ======== //
            log_info!("TODO: add2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("sub2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] - [rs2_high: rs2_low] ======== //
            log_info!("TODO: sub2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("sub2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] - [rs2_high: rs2_low] ======== //
            log_info!("TODO: sub2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_u32" , BIT32 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i32 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );
    itp.borrow_mut().def_insn("mul2_u64" , BIT64 | LITTLE_ENDIAN | INSN_USD, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== [rd_high: rd_low] = [rs1_high: rs1_low] * [rs2_high: rs2_low] ======== //
            log_info!("TODO: mul2_i64 rd_low, rd_high, rs1_low, rs1_high, rs2_low, rs2_high");
        }
    );

    itp.borrow_mut().def_insn("etr2_i32" , BIT32 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr2_i32 rd, rs1, rs2, pos");
        }
    );
    itp.borrow_mut().def_insn("etr2_i64" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1, pos, len ======== //
            log_info!("TODO: etr2_i64 rd, rs1, rs2, pos");
        }
    );

    itp.borrow_mut().def_insn("call" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== call ======== //
            log_info!("TODO: call ptr");
        }
    );
    itp.borrow_mut().def_insn("label" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== new label ======== //
            log_info!("TODO: label $label");
        }
    );
    itp.borrow_mut().def_insn("unlabel" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== unlabel ======== //
            log_info!("TODO: unlabel $label");
        }
    );
    itp.borrow_mut().def_insn("cond" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2) ======== //
            log_info!("TODO: cond rd, rs1, rs2, cc");
        }
    );
    itp.borrow_mut().def_insn("condval" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = (rs1 cc rs2 ? v1 : v2) ======== //
            log_info!("TODO: condval rd, rs1, rs2, cc, v1, v2");
        }
    );
    itp.borrow_mut().def_insn("jump" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== jump ======== //
            log_info!("TODO: jump $label");
        }
    );
    itp.borrow_mut().def_insn("branch" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== branch ======== //
            log_info!("TODO: branch $label, rs1, rs2, cc");
        }
    );
    itp.borrow_mut().def_insn("exit_tb" , BIT64 | LITTLE_ENDIAN | INSN_SIG, vec![OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== exit tb and return 0 to rd ======== //
            log_info!("TODO: exit_tb rd");
        }
    );



    Some(itp)
}

/// encode
pub fn evo_encode(insn: &mut Instruction, opr: Vec<Operand>) -> Instruction {
    if opr.len() == 0 {
        let mut res = insn.clone();
        res.is_applied = true;
        return res;
    }
    let mut opr = opr;
    // Check syms
    if insn.check_syms(opr.clone()) {
        // match opcode type kind and fill bytes by opreands
        match insn.opc.kind() {
            OpcodeKind::R(_, _) => {
                if opr.len() >= 3 {
                    // rs2: u5 -> 20->24
                    let rs2 = opr[2].val().get_byte(0);
                    insn.set_rs2(rs2);
                }
                if opr.len() >= 2 {
                    // rs1: u5 -> 15->19
                    let rs1 = opr[1].val().get_byte(0);
                    insn.set_rs1(rs1);
                }
                // rd: u5 -> 7->11
                let rd = opr[0].val().get_byte(0);
                insn.set_rd(rd);
            },
            OpcodeKind::I(_, _) => {
                // rd: u5 -> 7->11
                let rd = opr[0].val().get_byte(0);
                insn.set_rd(rd);
                // rs1: u5 -> 15->19
                let rs1 = opr[1].val().get_byte(0);
                insn.set_rs1(rs1);
                // imm: u12 -> 20->32
                let imm = opr[2].val().get_half(0);
                insn.set_imm_i(imm);
                // refresh imm
                opr.pop();
                opr.push(Operand::imm(Value::bit(12, imm as i128)));
            },
            OpcodeKind::S(_, _) => {
                // rs2: u5 -> 20->24
                let rs2 = opr[0].val().get_byte(0);
                insn.set_rs2(rs2);
                // rs1: u5 -> 15->19
                let rs1 = opr[1].val().get_byte(0);
                insn.set_rs1(rs1);
                // imm: S
                let imm = opr[2].val().get_half(0);
                insn.set_imm_s(imm);
                // refresh imm
                opr.pop();
                opr.push(Operand::imm(Value::bit(12, imm as i128)));
            },
            OpcodeKind::B(_, _) => {
                // rs2: u5 -> 20->24
                let rs2 = opr[0].val().get_byte(0);
                insn.set_rs2(rs2);
                // rs1: u5 -> 15->19
                let rs1 = opr[1].val().get_byte(0);
                insn.set_rs1(rs1);
                // imm: B
                let imm = opr[2].val().get_half(0);
                insn.set_imm_b(imm);
                // refresh imm
                opr.pop();
                opr.push(Operand::imm(Value::bit(12, imm as i128)));
            },
            OpcodeKind::U(_, _) => {
                // rd: u5 -> 7->11
                let rd = opr[0].val().get_byte(0);
                insn.set_rd(rd);
                // imm: U
                let imm = opr[1].val().get_word(0);
                insn.set_imm_u(imm);
                // refresh imm
                opr.pop();
                opr.push(Operand::imm(Value::bit(12, imm as i128)));
            },
            OpcodeKind::J(_, _) => {
                // rd: u5 -> 7->11
                let rd = opr[0].val().get_byte(0);
                insn.set_rd(rd);
                // imm: J
                let imm = opr[1].val().get_word(0);
                insn.set_imm_j(imm);
                // refresh imm
                opr.pop();
                opr.push(Operand::imm(Value::bit(20, imm as i128)));
            },
            _ => {
                log_warning!("Not support opcode type {} in arch {}", insn.opc.kind(), EVO_ARCH);
            },
        }
        // refresh status
        let mut res = insn.clone();
        res.opr = opr;
        res.is_applied = true;
        res
    } else {
        // Error
        log_error!("Encode operands failed: {} , check syms", insn.opc.name());
        // Revert
        Instruction::undef()
    }
}


/// decode from Value
pub fn evo_decode(value: Value) -> Instruction {
    let mut res = Instruction::undef();
    // 1. check scale
    if value.scale_sum() != 32 {
        log_error!("Invalid insn scale: {}", value.scale_sum());
        return res;
    }
    // 2. decode opc
    res.set_arch(&EVO_ARCH);
    res.code = value;
    let mut opr = vec![];
    match (res.opcode(), res.funct3(), res.funct7()) {
        // 2.1 R-Type
        (0b0110011, f3, f7) => {
            // Get oprands
            // a. rd
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.rd() as usize).borrow().clone());
            // b. rs1
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.rs1() as usize).borrow().clone());
            // c. rs2
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH, res.rs2() as usize).borrow().clone());
            // find insn
            match (f3, f7) {
                (0b000, 0b0000000) => res = Instruction::insn_pool_nget("add").borrow().clone(),
                (0b000, 0b0100000) => res = Instruction::insn_pool_nget("sub").borrow().clone(),
                (0b100, 0b0000000) => res = Instruction::insn_pool_nget("xor").borrow().clone(),
                (0b110, 0b0000000) => res = Instruction::insn_pool_nget("or").borrow().clone(),
                (0b111, 0b0000000) => res = Instruction::insn_pool_nget("and").borrow().clone(),
                (0b001, 0b0000000) => res = Instruction::insn_pool_nget("sll").borrow().clone(),
                (0b101, 0b0000000) => res = Instruction::insn_pool_nget("srl").borrow().clone(),
                (0b101, 0b0100000) => res = Instruction::insn_pool_nget("sra").borrow().clone(),
                (0b010, 0b0000000) => res = Instruction::insn_pool_nget("slt").borrow().clone(),
                (0b011, 0b0000000) => res = Instruction::insn_pool_nget("sltu").borrow().clone(),
                _ => {

                }
            }
        },
        // 2.2 I-Type
        (0b0010011, f3, 0b0000000) => {
            // Get oprands
            // a. rd
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH,res.rd() as usize).borrow().clone());
            // b. rs1
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH,res.rs1() as usize).borrow().clone());
            // c. imm
            opr.push(Operand::imm(Value::bit(12, res.imm_i() as i128)));
            // find insn
            match f3 {
                0b000 => res = Instruction::insn_pool_nget("addi").borrow().clone(),
                0b010 => res = Instruction::insn_pool_nget("slti").borrow().clone(),
                0b011 => res = Instruction::insn_pool_nget("sltiu").borrow().clone(),
                0b100 => res = Instruction::insn_pool_nget("xori").borrow().clone(),
                0b110 => res = Instruction::insn_pool_nget("ori").borrow().clone(),
                0b111 => res = Instruction::insn_pool_nget("andi").borrow().clone(),
                _ => {}
            }
        },
        // 2.3 S-Type
        (0b0100011, f3, 0b0000000) => {
            // Get oprands
            // a. rs1
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH,res.rs1() as usize).borrow().clone());
            // b. rs2
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH,res.rs2() as usize).borrow().clone());
            // c. imm
            opr.push(Operand::imm(Value::bit(12, res.imm_s() as i128)));
            // find insn
            match f3 {
                0b000 => res = Instruction::insn_pool_nget("sb").borrow().clone(),
                0b001 => res = Instruction::insn_pool_nget("sh").borrow().clone(),
                0b010 => res = Instruction::insn_pool_nget("sw").borrow().clone(),
                _ => {}
            }
        },
        // 2.4 B-Type
        (0b1100011, f3, 0b0000000) => {
            // Get oprands
            // a. rs1
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH,res.rs1() as usize).borrow().clone());
            // b. rs2
            opr.push(RegFile::reg_poolr_get(&EVO_ARCH,res.rs2() as usize).borrow().clone());
            // c. imm
            opr.push(Operand::imm(Value::bit(12, res.imm_b() as i128)));
            // find insn
            match f3 {
                0b000 => res = Instruction::insn_pool_nget("beq").borrow().clone(),
                0b001 => res = Instruction::insn_pool_nget("bne").borrow().clone(),
                _ => {}
            }
        }
        _ => {

        }

    }
    // 3. encode
    res.encode(opr)
}



#[cfg(test)]
mod evo_test {

    use super::*;
    use crate::core::cpu::CPUState;


    #[test]
    fn evo_itp() {
        let cpu = CPUState::init(&EVO_ARCH, &EVO_ARCH, None, None, None);
        cpu.set_nreg("t1", Value::i64(266));
        cpu.set_nreg("t2", Value::i64(-277));
        cpu.set_nreg("t3", Value::i64(65535));
        cpu.set_nreg("t4", Value::i64(65537));
        cpu.set_nreg("t5", Value::f64(3.1415926));
        cpu.mem_write(26, Value::i32(0x1ffff));
        println!("{}", cpu.pool_info());

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
        let insn49 = Instruction::from_string("mov_i32 t0, t1");
        let insn50 = Instruction::from_string("mov_i64 t0, t1");

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


        let insn63 = Instruction::from_string("etr_i32 t0, t4");
        let insn64 = Instruction::from_string("etr_i64 t0, t4");
        let insn65 = Instruction::from_string("etr_u32 t0, t4");
        let insn66 = Instruction::from_string("etr_u64 t0, t4");
        let insn67 = Instruction::from_string("etrh_i32 t0, t4");
        let insn68 = Instruction::from_string("etrl_i32 t0, t4");
        let insn69 = Instruction::from_string("etrh_i64 t0, t4");
        let insn70 = Instruction::from_string("etrl_i64 t0, t4");

        let insn71 = Instruction::from_string("truc_i32 t0, t4");
        let insn72 = Instruction::from_string("truc_i64 t0, t4");
        let insn73 = Instruction::from_string("conc_i64 t0, t4, t1");
        let insn74 = Instruction::from_string("conn_i64 t0, t4, t1");
        let insn75 = Instruction::from_string("swph_i32 t0, t4");
        let insn76 = Instruction::from_string("swph_i64 t0, t4");
        let insn77 = Instruction::from_string("swpw_i32 t0, t4");
        let insn78 = Instruction::from_string("swpw_i64 t0, t4");
        let insn79 = Instruction::from_string("swpd_i64 t0, t4");
        let insn80 = Instruction::from_string("depo_i32 t0, t4");
        let insn81 = Instruction::from_string("depo_i64 t0, t4");


        cpu.execute(&insn1);
        println!("{:<60} -> t0 = {}", insn1.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn2);
        println!("{:<60} -> t0 = {}", insn2.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn3);
        println!("{:<60} -> t0 = {}", insn3.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn4);
        println!("{:<60} -> t0 = {}", insn4.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn5);
        println!("{:<60} -> t0 = {}", insn5.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn6);
        println!("{:<60} -> t0 = {}", insn6.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn7);
        println!("{:<60} -> t0 = {}", insn7.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn8);
        println!("{:<60} -> t0 = {}", insn8.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn9);
        println!("{:<60} -> t0 = {}", insn9.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn10);
        println!("{:<60} -> t0 = {}", insn10.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn11);
        println!("{:<60} -> t0 = {}", insn11.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn12);
        println!("{:<60} -> t0 = {}", insn12.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn13);
        println!("{:<60} -> t0 = {}", insn13.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn14);
        println!("{:<60} -> t0 = {}", insn14.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn15);
        println!("{:<60} -> t0 = {}", insn15.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn16);

        println!("{:<60} -> t0 = {}", insn16.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn17);
        println!("{:<60} -> t0 = {}", insn17.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn18);
        println!("{:<60} -> t0 = {}", insn18.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn19);
        println!("{:<60} -> t0 = {}", insn19.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn20);
        println!("{:<60} -> t0 = {}", insn20.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn21);
        println!("{:<60} -> t0 = {}", insn21.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn22);
        println!("{:<60} -> t0 = {}", insn22.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn23);
        println!("{:<60} -> t0 = {}", insn23.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn24);
        println!("{:<60} -> t0 = {}", insn24.to_string(), cpu.get_nreg("t0").get_i64(0));

        cpu.execute(&insn25);
        println!("{:<60} -> t0 = {}", insn25.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn26);
        println!("{:<60} -> t0 = {}", insn26.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn27);
        println!("{:<60} -> t0 = {}", insn27.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn28);
        println!("{:<60} -> t0 = {}", insn28.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn29);
        println!("{:<60} -> t0 = {}", insn29.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn30);
        println!("{:<60} -> t0 = {}", insn30.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn31);
        println!("{:<60} -> t0 = {}", insn31.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn32);
        println!("{:<60} -> t0 = {}", insn32.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn33);
        println!("{:<60} -> t0 = {}", insn33.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn34);
        println!("{:<60} -> t0 = {}", insn34.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn35);
        println!("{:<60} -> t0 = {}", insn35.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn36);
        println!("{:<60} -> t0 = {}", insn36.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn37);
        println!("{:<60} -> t0 = {}", insn37.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn38);
        println!("{:<60} -> t0 = {}", insn38.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn39);
        println!("{:<60} -> t0 = {}", insn39.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn40);
        println!("{:<60} -> t0 = {}", insn40.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn41);
        println!("{:<60} -> t0 = {}", insn41.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn42);
        println!("{:<60} -> t0 = {}", insn42.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn43);
        println!("{:<60} -> t0 = {}", insn43.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn44);
        println!("{:<60} -> t0 = {}", insn44.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn45);
        println!("{:<60} -> t0 = {}", insn45.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn46);
        println!("{:<60} -> t0 = {}", insn46.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn47);
        println!("{:<60} -> t0 = {}", insn47.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn48);
        println!("{:<60} -> t0 = {}", insn48.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn49);
        println!("{:<60} -> t0 = {}", insn49.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn50);
        println!("{:<60} -> t0 = {}", insn50.to_string(), cpu.get_nreg("t0").get_i64(0));

        cpu.execute(&insn51);
        println!("{:<60} -> t0 = {}", insn51.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn52);
        println!("{:<60} -> t0 = {}", insn52.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn53);
        println!("{:<60} -> t0 = {}", insn53.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn54);
        println!("{:<60} -> t0 = {}", insn54.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn55);
        println!("{:<60} -> t0 = {}", insn55.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn56);
        println!("{:<60} -> t0 = {}", insn56.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn57);
        println!("{:<60} -> t0 = {}", insn57.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn58);
        println!("{:<60} -> t0 = {}", insn58.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn59);
        println!("{:<60} -> t0 = {}", insn59.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn60);
        println!("{:<60} -> t0 = {}", insn60.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn61);
        println!("{:<60} -> t0 = {}", insn61.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn62);
        println!("{:<60} -> t0 = {}", insn62.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn63);
        println!("{:<60} -> t0 = {}", insn63.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn64);
        println!("{:<60} -> t0 = {}", insn64.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn65);
        println!("{:<60} -> t0 = {}", insn65.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn66);
        println!("{:<60} -> t0 = {}", insn66.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn67);
        println!("{:<60} -> t0 = {}", insn67.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn68);
        println!("{:<60} -> t0 = {}", insn68.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn69);
        println!("{:<60} -> t0 = 0x{:02x}", insn69.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn70);
        println!("{:<60} -> t0 = 0x{:02x}", insn70.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn71);
        println!("{:<60} -> t0 = 0x{:02x}", insn71.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn72);
        println!("{:<60} -> t0 = 0x{:02x}", insn72.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn73);
        println!("{:<60} -> t0 = 0x{:02x}", insn73.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn74);
        println!("{:<60} -> t0 = 0x{:02x}", insn74.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn75);
        println!("{:<60} -> t0 = 0x{:02x}", insn75.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn76);
        println!("{:<60} -> t0 = 0x{:02x}", insn76.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn77);
        println!("{:<60} -> t0 = 0x{:02x}", insn77.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn78);
        println!("{:<60} -> t0 = 0x{:02x}", insn78.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn79);
        println!("{:<60} -> t0 = 0x{:02x}", insn79.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn80);
        println!("{:<60} -> t0 = 0x{:02x}", insn80.to_string(), cpu.get_nreg("t0").get_i64(0));
        cpu.execute(&insn81);
        println!("{:<60} -> t0 = 0x{:02x}", insn81.to_string(), cpu.get_nreg("t0").get_i64(0));

        // cpu.set_nreg("t0", Value::i32(56));
        // cpu.execute(&insn22);
        // println!("{:<60} -> mem = {}", insn22.to_string(), cpu.mem_read(26, 1).bin(0, 1, false));
        // cpu.set_nreg("t0", Value::i32(732));
        // cpu.execute(&insn23);
        // println!("{:<60} -> mem = {}", insn23.to_string(), cpu.mem_read(26, 1).bin(0, 2, false));
        // cpu.set_nreg("t0", Value::i32(-8739));
        // cpu.execute(&insn24);
        // println!("{:<60} -> mem = {}", insn24.to_string(), cpu.mem_read(26, 1).bin(0, 4, false));

        // cpu.execute(&insn31);
        // println!("{:<60} -> pc = {}, t0 = {}", insn31.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));
        // cpu.execute(&insn32);
        // println!("{:<60} -> pc = {}, t0 = {}", insn32.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));

        // cpu.execute(&insn33);
        // println!("{:<60} -> pc = {}, t0 = {}", insn33.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));
        // cpu.execute(&insn34);
        // println!("{:<60} -> pc = {}, t0 = {}", insn34.to_string(), cpu.get_pc(), cpu.get_nreg("t0").get_i32(0));

        // cpu.execute(&insn35);
        // println!("{:<60} -> status = {}", insn35.to_string(), cpu.status());
        // cpu.execute(&insn36);
        // println!("{:<60} -> status = {}", insn36.to_string(), cpu.status());
    }
}