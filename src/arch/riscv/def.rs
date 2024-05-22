

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::{log_error, log_warning};
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT32, LITTLE_ENDIAN};
use crate::core::val::Value;
use crate::core::op::{OpcodeKind, Operand, OPR_IMM, OPR_REG};
use crate::core::insn::{Instruction, RegFile};
use crate::core::itp::Interpreter;
use crate::core::mem::CPUThreadStatus;





// ============================================================================== //
//                                 RISCV-32
// ============================================================================== //

pub const RISCV32_ARCH: Arch = Arch::new(ArchKind::RISCV, BIT32 | LITTLE_ENDIAN, 33);


/// Insn temp and Reg and Interpreter Pool Init
pub fn riscv32_itp_init() -> Option<Rc<RefCell<Interpreter>>> {

    // 1. Init regs pool
    RegFile::def(&RISCV32_ARCH, "x0", Value::bit(5, 0), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x1", Value::bit(5, 1), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x2", Value::bit(5, 2), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x3", Value::bit(5, 3), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x4", Value::bit(5, 4), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x5", Value::bit(5, 5), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x6", Value::bit(5, 6), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x7", Value::bit(5, 7), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x8", Value::bit(5, 8), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x9", Value::bit(5, 9), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x10", Value::bit(5, 10), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x11", Value::bit(5, 11), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x12", Value::bit(5, 12), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x13", Value::bit(5, 13), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x14", Value::bit(5, 14), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x15", Value::bit(5, 15), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x16", Value::bit(5, 16), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x17", Value::bit(5, 17), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x18", Value::bit(5, 18), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x19", Value::bit(5, 19), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x20", Value::bit(5, 20), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x21", Value::bit(5, 21), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x22", Value::bit(5, 22), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x23", Value::bit(5, 23), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x24", Value::bit(5, 24), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x25", Value::bit(5, 25), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x26", Value::bit(5, 26), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x27", Value::bit(5, 27), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x28", Value::bit(5, 28), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x29", Value::bit(5, 29), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x30", Value::bit(5, 30), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "x31", Value::bit(5, 31), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&RISCV32_ARCH, "pc" , Value::bit(5, 32), BIT32 | LITTLE_ENDIAN);

    // 2. Init insns & insns interpreter
    let itp = Interpreter::def(&RISCV32_ARCH);
    // RISCV Instruction Format:                                           32|31  25|24 20|19 15|  |11  7|6    0|
    // Type: R                                [rd, rs1, rs2]                 |  f7  | rs2 | rs1 |f3|  rd |  op  |
    itp.borrow_mut().def_insn("add" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .000.... .0110011", 
        |cpu, insn| {
            // ======== rd = rs1 + rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Add rs1 and rs2
            let res = rs1.wrapping_add(rs2);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("sub" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .000.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 - rs2 ======== //
            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_i32(0);
            // 3. Sub rs1 and rs2
            let res = rs1.wrapping_sub(rs2);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("or"  , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .110.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Or rs1 and rs2
            let res = rs1 | rs2;
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("xor" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Xor rs1 and rs2
            let res = rs1 ^ rs2;
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("and" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .111.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. And rs1 and rs2
            let res = rs1 & rs2;
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("sll" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .001.... .0110011",
        |cpu, insn| {
            // ====== rd = rs1 << rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Sll rs1 and rs2
            let res = rs1.wrapping_shl(rs2);
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("srl" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0000000. ........ .101.... .0110011",
        |cpu, insn| {
            // ====== rd = rs1 >> rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(u32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_u32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Srl rs1 and rs2
            let res = rs1.wrapping_shr(rs2);
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(res));
        }
    );
    itp.borrow_mut().def_insn("sra" , BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_REG], "R", "0B0100000. ........ .101.... .0110011",
        |cpu, insn| {
            // ====== rd = rs1 >> rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get rs2(u32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_u32(0);
            // 3. Sra rs1 and rs2: Shift Right Arith
            let res = rs1.wrapping_shr(rs2);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
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
    itp.borrow_mut().def_insn("lb", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .000.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_byte(0);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
        }
    );
    itp.borrow_mut().def_insn("lh", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .001.... .0000011", 
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_half(0);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
        }
    );
    itp.borrow_mut().def_insn("lw", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .010.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Word
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_word(0);
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(val as i32));
        }
    );
    itp.borrow_mut().def_insn("lbu", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .100.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Byte
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_byte(0);
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(val as u32));
        }
    );
    itp.borrow_mut().def_insn("lhu", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "I", "0B........ ........ .101.... .0000011",
        |cpu, insn| {
            // ======== rd = [rs1 + imm] ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_i() as i32;
            // 3. Read Mem: Half
            let val = proc0.mem_read((rs1 + imm) as usize, 1).get_half(0);
            // 4. Set rd(u32)
            proc0.set_reg(insn.rd() as usize, Value::u32(val as u32));
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
            // System will do next part according to register `a7`(x17)
            proc0.set_status(CPUThreadStatus::Blocked);
        }
    );
    itp.borrow_mut().def_insn("ebreak", BIT32 | LITTLE_ENDIAN, vec![], "I", "0B00000000 0001.... .000.... .1110111",
        |cpu, _| {
            // ======== ebreak ======== //
            let proc0 = cpu.proc.borrow().clone();
            // Debugger will do next part according to register `a7`(x17)
            proc0.set_status(CPUThreadStatus::Blocked);
        }
    );
    // Type:S 
    itp.borrow_mut().def_insn("sb", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .000.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_byte(0) as i8;
            // 4. Write Mem: Byte
            proc0.mem_write((rs1 + imm) as usize, Value::i8(rs2));
        }
    );
    itp.borrow_mut().def_insn("sh", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .001.... .0100011",
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
    itp.borrow_mut().def_insn("sw", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG, OPR_IMM], "S", "0B........ ........ .010.... .0100011",
        |cpu, insn| {
            // ======== [rs1 + imm] = rs2 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get rs1(i32)
            let rs1 = proc0.get_reg(insn.rs1() as usize).get_i32(0);
            // 2. Get imm(i32)
            let imm = insn.imm_s() as i32;
            // 3. Get rs2(i32)
            let rs2 = proc0.get_reg(insn.rs2() as usize).get_word(0) as i32;
            // 4. Write Mem: Word
            proc0.mem_write((rs1 + imm) as usize, Value::i32(rs2));
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
    itp.borrow_mut().def_insn("lui", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_IMM], "U", "0B........ ........ ........ .0110111",
        |cpu, insn| {
            // ======== rd = imm << 12 ======== //

            let proc0 = cpu.proc.borrow().clone();
            // 1. Get imm(i32)
            let imm = insn.imm_u() as i32;
            // 2. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(imm << 12));
        }
    );
    itp.borrow_mut().def_insn("auipc", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_IMM], "U", "0B........ ........ ........ .0010111",
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
    itp.borrow_mut().def_insn("jal", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_IMM], "J", "0B........ ........ ........ .1101111",
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


    Some(itp)
}


/// encode
pub fn riscv32_encode(insn: &mut Instruction, opr: Vec<Operand>) -> Instruction {
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
                log_warning!("Not support opcode type {} in arch {}", insn.opc.kind(), RISCV32_ARCH);
            }
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
pub fn riscv32_decode(value: Value) -> Instruction {
    let mut res = Instruction::undef();
    // 1. check scale
    if value.scale_sum() != 32 {
        log_error!("Invalid insn scale: {}", value.scale_sum());
        return res;
    }
    // 2. decode opc
    res.set_arch(&RISCV32_ARCH);
    res.code = value;
    let mut opr = vec![];
    match (res.opcode(), res.funct3(), res.funct7()) {
        // 2.1 R-Type
        (0b0110011, f3, f7) => {
            // Get oprands
            // a. rd
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rd() as usize).borrow().clone());
            // b. rs1
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs1() as usize).borrow().clone());
            // c. rs2
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs2() as usize).borrow().clone());
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
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rd() as usize).borrow().clone());
            // b. rs1
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs1() as usize).borrow().clone());
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
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs1() as usize).borrow().clone());
            // b. rs2
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs2() as usize).borrow().clone());
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
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs1() as usize).borrow().clone());
            // b. rs2
            opr.push(RegFile::reg_poolr_get(&RISCV32_ARCH, res.rs2() as usize).borrow().clone());
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
mod riscv_test {

    use super::*;
    use crate::core::cpu::CPUState;


    #[test]
    fn rv32_itp() {
        let cpu = CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);
        cpu.set_nreg("x1", Value::i32(3));
        cpu.set_nreg("x2", Value::i32(5));
        cpu.set_nreg("x3", Value::i32(-32));
        cpu.mem_write(26, Value::i32(0x1ffff));
        println!("{}", cpu.pool_info());

        // R-Type Insns Test
        let insn1 = Instruction::from_string("add x0, x1, x2");
        let insn2 = Instruction::from_string("sub x0, x1, x2");
        let insn3 = Instruction::from_string("or x0, x1, x2");
        let insn4 = Instruction::from_string("xor x0, x1, x2");
        let insn5 = Instruction::from_string("and x0, x1, x2");
        let insn6 = Instruction::from_string("sltu x0, x1, x2");
        let insn7 = Instruction::from_string("srl x0, x1, x3");
        let insn8 = Instruction::from_string("sra x0, x1, x3");
        let insn9 = Instruction::from_string("sll x0, x1, x3");
        let insn10 = Instruction::from_string("slt x0, x1, x2");

        // I-Type Insns Test
        let insn11 = Instruction::from_string("addi x0, x1, 0xff 0f");
        let insn12 = Instruction::from_string("andi x0, x1, 2457");
        let insn13 = Instruction::from_string("ori x0, x1, 2457");
        let insn14 = Instruction::from_string("xori x0, x1, 2457");
        let insn15 = Instruction::from_string("slti x0, x1, 2");
        let insn16 = Instruction::from_string("sltiu x0, x1, 2");

        let insn17 = Instruction::from_string("lb x0, x1, 23");
        let insn18 = Instruction::from_string("lh x0, x1, 23");
        let insn19 = Instruction::from_string("lw x0, x1, 23");
        let insn20 = Instruction::from_string("lbu x0, x1, 23");
        let insn21 = Instruction::from_string("lhu x0, x1, 23");

        let insn22 = Instruction::from_string("sb x0, x1, 23");
        let insn23 = Instruction::from_string("sh x0, x1, 23");
        let insn24 = Instruction::from_string("sw x0, x1, 23");
        
        let insn34 = Instruction::from_string("jalr x0, x1, 23");
        let insn35 = Instruction::from_string("ecall");
        let insn36 = Instruction::from_string("ebreak");

        // B-Type Insns Test
        let insn25 = Instruction::from_string("beq x0, x1, 23");
        let insn26 = Instruction::from_string("bne x0, x1, 23");
        let insn27 = Instruction::from_string("blt x0, x1, 23");
        let insn28 = Instruction::from_string("bge x0, x1, 23");
        let insn29 = Instruction::from_string("bltu x0, x1, 23");
        let insn30 = Instruction::from_string("bgeu x0, x1, 23");

        // U-Type Insns Test
        let insn31 = Instruction::from_string("lui x0, 255");
        let insn32 = Instruction::from_string("auipc x0, 255");
        // J-Type Insns Test
        let insn33 = Instruction::from_string("jal x0, 23");

        cpu.execute(&insn1);
        println!("{:<60} -> x0 = {}", insn1.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn2);
        println!("{:<60} -> x0 = {}", insn2.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn3);
        println!("{:<60} -> x0 = {}", insn3.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn4);
        println!("{:<60} -> x0 = {}", insn4.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn5);
        println!("{:<60} -> x0 = {}", insn5.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn6);
        println!("{:<60} -> x0 = {}", insn6.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn7);
        println!("{:<60} -> x0 = {}", insn7.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn8);
        println!("{:<60} -> x0 = {}", insn8.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn9);
        println!("{:<60} -> x0 = {}", insn9.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn10);
        println!("{:<60} -> x0 = {}", insn10.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn11);
        println!("{:<60} -> x0 = {}", insn11.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn12);
        println!("{:<60} -> x0 = {}", insn12.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn13);
        println!("{:<60} -> x0 = {}", insn13.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn14);
        println!("{:<60} -> x0 = {}", insn14.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn15);
        println!("{:<60} -> x0 = {}", insn15.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn16);
        println!("{:<60} -> x0 = {}", insn16.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn17);
        println!("{:<60} -> x0 = {}", insn17.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn18);
        println!("{:<60} -> x0 = {}", insn18.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn19);
        println!("{:<60} -> x0 = {}", insn19.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn20);
        println!("{:<60} -> x0 = {}", insn20.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn21);
        println!("{:<60} -> x0 = {}", insn21.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.set_nreg("x0", Value::i32(56));
        cpu.execute(&insn22);
        println!("{:<60} -> mem = {}", insn22.to_string(), cpu.mem_read(26, 1).bin(0, 1, false));
        cpu.set_nreg("x0", Value::i32(732));
        cpu.execute(&insn23);
        println!("{:<60} -> mem = {}", insn23.to_string(), cpu.mem_read(26, 1).bin(0, 2, false));
        cpu.set_nreg("x0", Value::i32(-8739));
        cpu.execute(&insn24);
        println!("{:<60} -> mem = {}", insn24.to_string(), cpu.mem_read(26, 1).bin(0, 4, false));

        cpu.execute(&insn25);
        println!("{:<60} -> pc = {}", insn25.to_string(), cpu.get_pc());
        cpu.execute(&insn26);
        println!("{:<60} -> pc = {}", insn26.to_string(), cpu.get_pc());
        cpu.execute(&insn27);
        println!("{:<60} -> pc = {}", insn27.to_string(), cpu.get_pc());
        cpu.execute(&insn28);
        println!("{:<60} -> pc = {}", insn28.to_string(), cpu.get_pc());
        cpu.execute(&insn29);
        println!("{:<60} -> pc = {}", insn29.to_string(), cpu.get_pc());
        cpu.execute(&insn30);
        println!("{:<60} -> pc = {}", insn30.to_string(), cpu.get_pc());

        cpu.execute(&insn31);
        println!("{:<60} -> pc = {}, x0 = {}", insn31.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn32);
        println!("{:<60} -> pc = {}, x0 = {}", insn32.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn33);
        println!("{:<60} -> pc = {}, x0 = {}", insn33.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn34);
        println!("{:<60} -> pc = {}, x0 = {}", insn34.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn35);
        println!("{:<60} -> status = {}", insn35.to_string(), cpu.status());
        cpu.execute(&insn36);
        println!("{:<60} -> status = {}", insn36.to_string(), cpu.status());
    }
}