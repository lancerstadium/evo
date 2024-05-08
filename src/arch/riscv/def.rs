

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT32, LITTLE_ENDIAN};
use crate::ir::val::Value;
use crate::ir::insn::Instruction;
use crate::ir::itp::Interpreter;
use crate::ir::mem::CPUThreadStatus;




pub const RISCV32_ARCH: Arch = Arch::new(ArchKind::RISCV, BIT32 | LITTLE_ENDIAN, 33);



/// Insn temp and Reg and Interpreter Pool Init
pub fn riscv32_itp_init() -> Option<Rc<RefCell<Interpreter>>> {

    // 2. Init regs pool
    Instruction::reg("x0", Value::bit(5, 0));
    Instruction::reg("x1", Value::bit(5, 1));
    Instruction::reg("x2", Value::bit(5, 2));
    Instruction::reg("x3", Value::bit(5, 3));
    Instruction::reg("x4", Value::bit(5, 4));
    Instruction::reg("x5", Value::bit(5, 5));
    Instruction::reg("x6", Value::bit(5, 6));
    Instruction::reg("x7", Value::bit(5, 7));
    Instruction::reg("x8", Value::bit(5, 8));
    Instruction::reg("x9", Value::bit(5, 9));
    Instruction::reg("x10", Value::bit(5, 10));
    Instruction::reg("x11", Value::bit(5, 11));
    Instruction::reg("x12", Value::bit(5, 12));
    Instruction::reg("x13", Value::bit(5, 13));
    Instruction::reg("x14", Value::bit(5, 14));
    Instruction::reg("x15", Value::bit(5, 15));
    Instruction::reg("x16", Value::bit(5, 16));
    Instruction::reg("x17", Value::bit(5, 17));
    Instruction::reg("x18", Value::bit(5, 18));
    Instruction::reg("x19", Value::bit(5, 19));
    Instruction::reg("x20", Value::bit(5, 20));
    Instruction::reg("x21", Value::bit(5, 21));
    Instruction::reg("x22", Value::bit(5, 22));
    Instruction::reg("x23", Value::bit(5, 23));
    Instruction::reg("x24", Value::bit(5, 24));
    Instruction::reg("x25", Value::bit(5, 25));
    Instruction::reg("x26", Value::bit(5, 26));
    Instruction::reg("x27", Value::bit(5, 27));
    Instruction::reg("x28", Value::bit(5, 28));
    Instruction::reg("x29", Value::bit(5, 29));
    Instruction::reg("x30", Value::bit(5, 30));
    Instruction::reg("x31", Value::bit(5, 31));
    Instruction::reg("pc" , Value::bit(5, 32));

    // 3. Init insns & insns interpreter
    let itp = Interpreter::def(&RISCV32_ARCH);
    // RISCV Instruction Format:                                           32|31  25|24 20|19 15|  |11  7|6    0|
    // Type: R                                [rd, rs1, rs2]                 |  f7  | rs2 | rs1 |f3|  rd |  op  |
    itp.borrow_mut().def_insn("add" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .000.... .0110011", 
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
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("sub" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0100000. ........ .000.... .0110011",
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
            // 4. Set rd(i32)
            proc0.set_reg(insn.rd() as usize, Value::i32(res));
        }
    );
    itp.borrow_mut().def_insn("or"  , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .110.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 | rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
    itp.borrow_mut().def_insn("xor" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .100.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 ^ rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
    itp.borrow_mut().def_insn("and" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .111.... .0110011",
        |cpu, insn| {
            // ======== rd = rs1 & rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
    itp.borrow_mut().def_insn("sll" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .001.... .0110011",
        |cpu, insn| {
            // ====== rd = rs1 << rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
    itp.borrow_mut().def_insn("srl" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0000000. ........ .101.... .0110011",
        |cpu, insn| {
            // ====== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
    itp.borrow_mut().def_insn("sra" , BIT32 | LITTLE_ENDIAN, vec![1, 1, 1], "R", "0B0100000. ........ .101.... .0110011",
        |cpu, insn| {
            // ====== rd = rs1 >> rs2 ======== //
            if !insn.is_applied {
                log_warning!("Insn not applied: {}", insn);
                return;
            }
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
            // System will do next part according to register `a7`(x17)
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
            // Debugger will do next part according to register `a7`(x17)
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
mod riscv_test {

    use super::*;
    use crate::ir::cpu::CPUState;


    #[test]
    fn rv32_itp() {
        let cpu = CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);
        cpu.set_nreg("x1", Value::i32(3));
        cpu.set_nreg("x2", Value::i32(5));
        cpu.set_nreg("x3", Value::i32(-32));
        cpu.write_mem(26, Value::i32(0x1ffff));

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
        let insn11 = Instruction::from_string("addi x0, x1, 2457");
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
        println!("{:<50} -> x0 = {}", insn1.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn2);
        println!("{:<50} -> x0 = {}", insn2.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn3);
        println!("{:<50} -> x0 = {}", insn3.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn4);
        println!("{:<50} -> x0 = {}", insn4.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn5);
        println!("{:<50} -> x0 = {}", insn5.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn6);
        println!("{:<50} -> x0 = {}", insn6.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn7);
        println!("{:<50} -> x0 = {}", insn7.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn8);
        println!("{:<50} -> x0 = {}", insn8.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn9);
        println!("{:<50} -> x0 = {}", insn9.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn10);
        println!("{:<50} -> x0 = {}", insn10.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn11);
        println!("{:<50} -> x0 = {}", insn11.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn12);
        println!("{:<50} -> x0 = {}", insn12.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn13);
        println!("{:<50} -> x0 = {}", insn13.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn14);
        println!("{:<50} -> x0 = {}", insn14.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn15);
        println!("{:<50} -> x0 = {}", insn15.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn16);
        println!("{:<50} -> x0 = {}", insn16.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn17);
        println!("{:<50} -> x0 = {}", insn17.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn18);
        println!("{:<50} -> x0 = {}", insn18.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn19);
        println!("{:<50} -> x0 = {}", insn19.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn20);
        println!("{:<50} -> x0 = {}", insn20.to_string(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn21);
        println!("{:<50} -> x0 = {}", insn21.to_string(), cpu.get_nreg("x0").get_i32(0));

        cpu.set_nreg("x0", Value::i32(56));
        cpu.execute(&insn22);
        println!("{:<50} -> mem = {}", insn22.to_string(), cpu.read_mem(26, 1).bin(0, 1, false));
        cpu.set_nreg("x0", Value::i32(732));
        cpu.execute(&insn23);
        println!("{:<50} -> mem = {}", insn23.to_string(), cpu.read_mem(26, 1).bin(0, 2, false));
        cpu.set_nreg("x0", Value::i32(-8739));
        cpu.execute(&insn24);
        println!("{:<50} -> mem = {}", insn24.to_string(), cpu.read_mem(26, 1).bin(0, 4, false));

        cpu.execute(&insn25);
        println!("{:<50} -> pc = {}", insn25.to_string(), cpu.get_pc());
        cpu.execute(&insn26);
        println!("{:<50} -> pc = {}", insn26.to_string(), cpu.get_pc());
        cpu.execute(&insn27);
        println!("{:<50} -> pc = {}", insn27.to_string(), cpu.get_pc());
        cpu.execute(&insn28);
        println!("{:<50} -> pc = {}", insn28.to_string(), cpu.get_pc());
        cpu.execute(&insn29);
        println!("{:<50} -> pc = {}", insn29.to_string(), cpu.get_pc());
        cpu.execute(&insn30);
        println!("{:<50} -> pc = {}", insn30.to_string(), cpu.get_pc());

        cpu.execute(&insn31);
        println!("{:<50} -> pc = {}, x0 = {}", insn31.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn32);
        println!("{:<50} -> pc = {}, x0 = {}", insn32.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn33);
        println!("{:<50} -> pc = {}, x0 = {}", insn33.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));
        cpu.execute(&insn34);
        println!("{:<50} -> pc = {}, x0 = {}", insn34.to_string(), cpu.get_pc(), cpu.get_nreg("x0").get_i32(0));

        cpu.execute(&insn35);
        println!("{:<50} -> status = {}", insn35.to_string(), cpu.status());
        cpu.execute(&insn36);
        println!("{:<50} -> status = {}", insn36.to_string(), cpu.status());
    }
}