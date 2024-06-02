

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;


use crate::{log_warning, log_error};
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT16, BIT32, BIT64, BIT8, LITTLE_ENDIAN};
use crate::core::val::Value;
use crate::core::op::{OpcodeKind, Operand, OperandKind, OPR_IMM, OPR_MEM, OPR_REG, REG_OFF8};
use crate::core::insn::{Instruction, RegFile, INSN_SIG, INSN_USD};
use crate::core::itp::Interpreter;
use crate::core::mem::CPUThreadStatus;




// ============================================================================== //
//                             x86::def::arch
// ============================================================================== //


/// `i386`
/// ### Registers
/// 
/// ```txt
/// ┌────────┬───────────┬───────────┬──────────┬─────────┐
/// │ Encode │ r32[31:0] │ r16[15:0] │ r8[15:8] │ r8[7:0] │
/// ├────────┼───────────┼───────────┼──────────┼─────────┤
/// │  000   │    eax    │    ax     │    ah    │   al    │  
/// │  001   │    ecx    │    cx     │    ch    │   cl    │
/// │  010   │    edx    │    dx     │    dh    │   dl    │
/// │  011   │    ebx    │    bx     │    bh    │   bl    │
/// │  100   │    esp    │    sp     │    --    │   --    │
/// │  101   │    ebp    │    bp     │    --    │   --    │
/// │  110   │    esi    │    si     │    --    │   --    │
/// │  111   │    edi    │    di     │    --    │   --    │
/// └────────┴───────────┴───────────┴──────────┴─────────┘
/// ```
pub const X86_ARCH: Arch = Arch::new(ArchKind::X86, BIT32 | LITTLE_ENDIAN, 8);

/// `amd64`
/// ### Registers
/// 
/// ```txt
/// ┌────────┬────────┬────────┬────────┬────────┬───────┐
/// │ Encode │ [63:0] │ [31:0] │ [15:0] │ [15:8] │ [7:0] │
/// ├────────┼────────┼────────┼────────┼────────┼───────┤
/// │  0000  │  rax   │  eax   │   ax   │   ah   │  al   │  
/// │  0001  │  rcx   │  ecx   │   cx   │   ch   │  cl   │
/// │  0010  │  rdx   │  edx   │   dx   │   dh   │  dl   │
/// │  0011  │  rbx   │  ebx   │   bx   │   bh   │  bl   │
/// │  0100  │  rsp   │  esp   │   sp   │   --   │  --   │
/// │  0101  │  rbp   │  ebp   │   bp   │   --   │  --   │
/// │  0110  │  rsi   │  esi   │   si   │   --   │  --   │
/// │  0111  │  rdi   │  edi   │   di   │   --   │  --   │
/// │  1000  │  r8    │  r8d   │  r8h   │  r8b   │  r8l  │
/// │  1001  │  r9    │  r9d   │  r9h   │  r9b   │  r9l  │
/// │  1010  │  r10   │  r10d  │  r10h  │  r10b  │ r10l  │
/// │  1011  │  r11   │  r11d  │  r11h  │  r11b  │ r11l  │
/// │  1100  │  r12   │  r12d  │  r12h  │  r12b  │ r12l  │
/// │  1101  │  r13   │  r13d  │  r13h  │  r13b  │ r13l  │
/// │  1110  │  r14   │  r14d  │  r14h  │  r14b  │ r14l  │
/// │  1111  │  r15   │  r15d  │  r15h  │  r15b  │ r15l  │
/// └────────┴────────┴────────┴────────┴────────┴───────┘
/// ```
pub const X86_64_ARCH: Arch = Arch::new(ArchKind::X86, BIT64 | LITTLE_ENDIAN, 16);

// ============================================================================== //
//                          evo::def::interpreter
// ============================================================================== //

/// Insn temp and Reg and Interpreter Pool Init
pub fn x86_itp_init() -> Option<Rc<RefCell<Interpreter>>> {

    // 1. Init regs pool
    RegFile::def(&X86_ARCH, "eax", Value::bit(3, 0), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "ecx", Value::bit(3, 1), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "edx", Value::bit(3, 2), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "ebx", Value::bit(3, 3), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "esp", Value::bit(3, 4), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "ebp", Value::bit(3, 5), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "esi", Value::bit(3, 6), BIT32 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "edi", Value::bit(3, 7), BIT32 | LITTLE_ENDIAN);

    RegFile::def(&X86_ARCH, "ax", Value::bit(3, 0), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "cx", Value::bit(3, 1), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "dx", Value::bit(3, 2), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "bx", Value::bit(3, 3), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "sp", Value::bit(3, 4), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "bp", Value::bit(3, 5), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "si", Value::bit(3, 6), BIT16 | LITTLE_ENDIAN);
    RegFile::def(&X86_ARCH, "di", Value::bit(3, 7), BIT16 | LITTLE_ENDIAN);

    RegFile::def(&X86_ARCH, "ah", Value::bit(3, 0), BIT8 | LITTLE_ENDIAN | REG_OFF8);
    RegFile::def(&X86_ARCH, "ch", Value::bit(3, 1), BIT8 | LITTLE_ENDIAN | REG_OFF8);
    RegFile::def(&X86_ARCH, "dh", Value::bit(3, 2), BIT8 | LITTLE_ENDIAN | REG_OFF8);
    RegFile::def(&X86_ARCH, "bh", Value::bit(3, 3), BIT8 | LITTLE_ENDIAN | REG_OFF8);

    RegFile::def(&X86_ARCH, "al", Value::bit(3, 0), BIT8 | LITTLE_ENDIAN | REG_OFF8);
    RegFile::def(&X86_ARCH, "cl", Value::bit(3, 1), BIT8 | LITTLE_ENDIAN | REG_OFF8);
    RegFile::def(&X86_ARCH, "dl", Value::bit(3, 2), BIT8 | LITTLE_ENDIAN | REG_OFF8);
    RegFile::def(&X86_ARCH, "bl", Value::bit(3, 3), BIT8 | LITTLE_ENDIAN | REG_OFF8);

    // 2. Init insns & insns interpreter
    let itp = Interpreter::def(&X86_ARCH);

    itp.borrow_mut().def_insn("mov", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "", 
        |cpu, insn| {
            
        }
    );

    Some(itp)
}



/// encode
pub fn x86_encode(insn: &mut Instruction, opr: Vec<Operand>) -> Instruction {
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
        let syms = opr.iter().map(|x| x.sym()).collect::<Vec<_>>();
        match insn.opc.kind() {
            OpcodeKind::X(_, _) => {
                match (insn.name(), syms.as_slice()) {
                    ("mov", [OPR_REG, OPR_REG]) => {
                        if opr[0].is_8bit() {
                            code.push(0x88);
                            let rm8 = opr[0].val().get_byte(0);
                            let r8  = opr[1].val().get_byte(0);
                            code.push(0b11_000_000 | rm8 << 3 | r8);
                        } else {
                            code.push(0x89);
                            let rm = opr[0].val().get_byte(0);
                            let r  = opr[1].val().get_byte(0);
                            code.push(0b11_000_000 | rm << 3 | r);
                        }
                    },
                    ("mov", [OPR_MEM, OPR_REG]) => {
                        if opr[1].is_8bit() {
                            code.push(0x88);
                            let rm8 = opr[0].get_mem().0 as u8;
                            let r8  = opr[1].val().get_byte(0);
                            code.push(0b11_000_000 | rm8 << 3 | r8);
                        } else {
                            code.push(0x89);
                            let rm = opr[0].get_mem().0 as u8;
                            let r  = opr[1].val().get_byte(0);
                            code.push(0b11_000_000 | rm << 3 | r);
                        }
                    },
                    ("mov", [OPR_REG, OPR_MEM]) => {
                        if opr[0].is_8bit() {
                            code.push(0x8a);
                            let r8 = opr[0].val().get_byte(0);
                            let rm8  = opr[1].get_mem().0 as u8;
                            code.push(0b11_000_000 | rm8 << 3 | r8);
                        } else {
                            code.push(0x8b);
                            let r = opr[0].val().get_byte(0);
                            let rm  = opr[1].get_mem().0 as u8;
                            code.push(0b11_000_000 | rm << 3 | r);
                        }
                    },
                    ("mov", [OPR_REG, OPR_IMM]) => {
                        if opr[0].is_8bit() {
                            code.push(0xb0);
                            let r8 = opr[0].val().get_byte(0);
                            code.push(0b11_000_000 | 0b000 << 3 | r8);
                            let imm8 = opr[1].val().get_byte(0);
                            new_opr[1] = Operand::imm(Value::u8(imm8));
                            code.push(imm8);
                        } else {
                            code.push(0xb8);
                            let r = opr[0].val().get_byte(0);
                            code.push(0b11_000_000 | 0b000 << 3 | r);
                            match opr[0].reg_scale() {
                                16 => {
                                    let imm16 = opr[1].val().get_half(0);
                                    new_opr[1] = Operand::imm(Value::u16(imm16));
                                    code.push(imm16 as u8);
                                    code.push((imm16 >> 8) as u8);
                                },
                                32 => {
                                    let imm32 = opr[1].val().get_word(0);
                                    new_opr[1] = Operand::imm(Value::u32(imm32));
                                    code.push(imm32 as u8);
                                    code.push((imm32 >> 8) as u8);
                                    code.push((imm32 >> 16) as u8);
                                    code.push((imm32 >> 24) as u8);
                                },
                                64 => {
                                    let imm64 = opr[1].val().get_dword(0);
                                    new_opr[1] = Operand::imm(Value::u64(imm64));
                                    code.push(imm64 as u8);
                                    code.push((imm64 >> 8) as u8);
                                    code.push((imm64 >> 16) as u8);
                                    code.push((imm64 >> 24) as u8);
                                    code.push((imm64 >> 32) as u8);
                                    code.push((imm64 >> 40) as u8);
                                    code.push((imm64 >> 48) as u8);
                                    code.push((imm64 >> 56) as u8);
                                },
                                _ => {
                                    log_error!("Not support scale {}", opr[0].reg_scale());
                                },
                            }
                        }
                    },
                    ("mov", [OPR_MEM, OPR_IMM]) => {
                        let rm_reg_scale = opr[0].get_mem_base().reg_scale();
                        if rm_reg_scale == 8 {
                            code.push(0xc6);
                            let rm8 = opr[0].get_mem().0 as u8;
                            code.push(0b11_000_000 | rm8 << 3);
                            let imm8 = opr[1].val().get_byte(0);
                            new_opr[1] = Operand::imm(Value::u8(imm8));
                            code.push(imm8);
                        } else {
                            code.push(0xc7);
                            let rm = opr[0].get_mem().0 as u8;
                            code.push(0b11_000_000 | rm << 3);
                            match rm_reg_scale {
                                16 => {
                                    let imm16 = opr[1].val().get_half(0);
                                    new_opr[1] = Operand::imm(Value::u16(imm16));
                                    code.push(imm16 as u8);
                                    code.push((imm16 >> 8) as u8);
                                },
                                32 => {
                                    let imm32 = opr[1].val().get_word(0);
                                    new_opr[1] = Operand::imm(Value::u32(imm32));
                                    code.push(imm32 as u8);
                                    code.push((imm32 >> 8) as u8);
                                    code.push((imm32 >> 16) as u8);
                                    code.push((imm32 >> 24) as u8);
                                },
                                64 => {
                                    let imm64 = opr[1].val().get_dword(0);
                                    new_opr[1] = Operand::imm(Value::u64(imm64));
                                    code.push(imm64 as u8);
                                    code.push((imm64 >> 8) as u8);
                                    code.push((imm64 >> 16) as u8);
                                    code.push((imm64 >> 24) as u8);
                                    code.push((imm64 >> 32) as u8);
                                    code.push((imm64 >> 40) as u8);
                                    code.push((imm64 >> 48) as u8);
                                    code.push((imm64 >> 56) as u8);
                                },
                                _ => {
                                    log_error!("Not support scale {}", opr[0].reg_scale());
                                },
                            }
                        }
                    },
                    _ => {
                        is_applied = false;
                        log_error!("Encode operands not implemented: {} {}", insn.opc.name(), syms.iter().map(|x| OperandKind::sym_str(*x)).collect::<Vec<_>>().join(", "));
                    },
                }
            },
            _ => {
                is_applied = false;
                log_warning!("Not support opcode type {} in arch {}", insn.opc.kind(), X86_ARCH);
            },
        }
        // refresh status
        let mut res = insn.clone();
        res.opr = new_opr;
        res.is_applied = is_applied;
        res.code = Value::array_u8(RefCell::new(code));
        res
    } else {
        // Error
        log_error!("Encode operands failed: {} , check syms", insn.opc.name());
        // Revert
        Instruction::undef()
    }
}


/// decode from Value
pub fn x86_decode(value: Value) -> Instruction {
    let mut res = Instruction::undef();
    // 1. check scale
    if value.scale_sum() != 32 {
        log_error!("Invalid insn scale: {}", value.scale_sum());
        return res;
    }
    // 2. decode opc
    res.set_arch(&X86_ARCH);
    res.code = value;
    let mut opr = vec![];
    // 3. deal with prefix

    // 4. deal with opcode

    // encode
    res.encode(opr)
}

#[cfg(test)]
mod x86_test {

    use super::*;
    use crate::core::cpu::CPUState;


    #[test]
    fn x86_itp() {
        let cpu = CPUState::init(&X86_ARCH, &X86_ARCH, None, None, None);
        cpu.set_nreg("eax", Value::i32(12));
        cpu.set_nreg("ebx", Value::i32(3));
        cpu.mem_write(26, Value::i32(0x1ffff));

        println!("{}", RegFile::reg_pool_info(&X86_ARCH));

        let insn1 = Instruction::from_string(&X86_ARCH, "mov [ah], 0x0");
        println!("code: {}", insn1.code);
        println!("{} -> eax: {}", insn1.to_string(), cpu.get_nreg("eax").get_i32(0));
    }

}