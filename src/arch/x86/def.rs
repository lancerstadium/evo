

use std::ops::Add;
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
pub const X86_ARCH: Arch = Arch::new(ArchKind::X86, BIT32 | LITTLE_ENDIAN, 8, ["byte ptr", "word ptr", "dword ptr", "qword ptr"]);

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
pub const X86_64_ARCH: Arch = Arch::new(ArchKind::X86, BIT64 | LITTLE_ENDIAN, 16, ["byte ptr", "word ptr", "dword ptr", "qword ptr"]);

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

    itp.borrow_mut().def_insn("add", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x00",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("or", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x08",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("adc", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x10",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("sbb", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x18",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("and", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x20",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("sub", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x28",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("xor", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x30",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("cmp", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x38",
        |cpu, insn| {
            
        }
    );
    // pattern 2
    itp.borrow_mut().def_insn("inc", BIT32 | LITTLE_ENDIAN, vec![OPR_REG], "X", "0x40",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("dec", BIT32 | LITTLE_ENDIAN, vec![OPR_REG], "X", "0x48",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("push", BIT32 | LITTLE_ENDIAN, vec![OPR_REG], "X", "0x50",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("pop", BIT32 | LITTLE_ENDIAN, vec![OPR_REG], "X", "0x58",
        |cpu, insn| {
            
        }
    );
    // Pattern 3
    itp.borrow_mut().def_insn("jo", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x70",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jno", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x71",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jb", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x72",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jnb", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x73",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("je", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x74",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jne", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x75",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jbe", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x76",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("ja", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x77",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("js", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x78",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jns", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x79",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jpe", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x7a",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jpo", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x7b",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jl", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x7c",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jge", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x7d",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jle", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x7e",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("jg", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x7f",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("test", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG], "X", "0x84",
        |cpu, insn| {
            
        }
    );
    itp.borrow_mut().def_insn("xchg", BIT32 | LITTLE_ENDIAN, vec![OPR_REG, OPR_REG], "X", "0x86",
        |cpu, insn| {
            
        }
    );
    // 
    itp.borrow_mut().def_insn("mov", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM, OPR_REG | OPR_MEM | OPR_IMM], "X", "0x88", 
        |cpu, insn| {
            
        }
    );
    // 
    itp.borrow_mut().def_insn("lea", BIT32 | LITTLE_ENDIAN, vec![OPR_REG | OPR_MEM], "X", "0x8d",
        |cpu, insn| {
            
        }
    );

    itp.borrow_mut().def_insn("nop", BIT32 | LITTLE_ENDIAN, vec![], "X", "0x90",
        |cpu, insn| {
            
        }
    );

    Some(itp)
}



/// encode
/// site: http://ref.x86asm.net/coder32.html
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
        code.extend_from_slice(&insn.code.get_bin(0));
        let syms = opr.iter().map(|x| x.sym()).collect::<Vec<_>>();
        match insn.opc.kind() {
            OpcodeKind::X(_, _) => {
                let pattern: u32 = match insn.name() {         // pre operate
                    "add" | "or" | "adc" | "sbb" | "and" | "sub" | "xor" | "cmp" => {
                        1
                    },
                    "inc" | "dec" | "push" | "pop" => {
                        2
                    },
                    "jo" | "jno" | "jb" | "jae" | "je" | "jne" | "jbe" | "ja" | "js" | "jns" | "jp" | "jpe" | "jnp" | "jpo" | "jl" | "jge" | "jle" | "jg" => {
                        3
                    },
                    _ => {
                        0
                    }
                };
                match (pattern, syms.as_slice()) {
                    (0, _) => {},                  // nothing
                    (1, [OPR_REG, OPR_REG]) => {    // [r/m8 r8]  [r/m16/32 r16/32]
                        if opr[0].is_8bit() {
                            let rm8 = opr[0].val().get_byte(0);
                            let r8  = opr[1].val().get_byte(0);
                            code.push(0b11_000_000 | r8 << 3 | rm8);
                        } else {
                            if let Some(last) = code.last_mut() { *last += 1; }
                            let rm = opr[0].val().get_byte(0);
                            let r  = opr[1].val().get_byte(0);
                            code.push(0b11_000_000 | r << 3 | rm);
                        }
                    },
                    (1, [OPR_MEM, OPR_REG]) => {    // [r/m8 r8]  [r/m16/32 r16/32]
                        if opr[1].is_8bit() {
                            let rm8 = opr[0].get_mem().0 as u8;
                            let r8  = opr[1].val().get_byte(0);
                            code.push(0b00_000_000 | r8 << 3 | rm8);
                        } else {
                            if let Some(last) = code.last_mut() { *last += 1; }
                            let rm = opr[0].get_mem().0 as u8;
                            let r  = opr[1].val().get_byte(0);
                            code.push(0b00_000_000 | r << 3 | rm);
                        }
                    },
                    (1, [OPR_REG, OPR_MEM]) => {    // [r8 r/m8]  [r16/32 r/m16/32]
                        if opr[0].is_8bit() {
                            if let Some(last) = code.last_mut() { *last += 2; }
                            let r8  = opr[0].val().get_byte(0);
                            let rm8 = opr[1].get_mem().0 as u8;
                            code.push(0b11_000_000 | r8 << 3 | rm8);
                        } else {
                            if let Some(last) = code.last_mut() { *last += 3; }
                            let r  = opr[0].val().get_byte(0);
                            let rm = opr[1].get_mem().0 as u8;
                            code.push(0b11_000_000 | r << 3 | rm);
                        }
                    },
                    (1, [OPR_REG, OPR_IMM]) => {    // [AL imm8]  [eAX imm16/32]
                        if opr[0].is_8bit() {
                            if let Some(last) = code.last_mut() { *last += 4; }
                            let r8 = opr[0].val().get_byte(0);
                            code.push(0b11_000_000 | r8 << 3 | 0b000);
                            let imm8 = opr[1].val().get_byte(0);
                            new_opr[1] = Operand::imm(Value::u8(imm8));
                            code.push(imm8);
                        } else {
                            if let Some(last) = code.last_mut() { *last += 5; }
                            let r = opr[0].val().get_byte(0);
                            code.push(0b11_000_000 | r << 3 | 0b000);
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
                    (2, [OPR_REG]) => {             // [r16/32]
                        if let Some(last) = code.last_mut() { *last += opr[0].val().get_byte(0); }
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

        // println!("{}", RegFile::reg_pool_info(&X86_ARCH));

        let insn1 = Instruction::from_string(&X86_ARCH, "add ecx, eax");
        let insn2 = Instruction::from_string(&X86_ARCH, "add [eax], eax");
        let insn3 = Instruction::from_string(&X86_ARCH, "add ax, [bx]");
        let insn4 = Instruction::from_string(&X86_ARCH, "add ax, 0x1ffff"); 
        println!("{:20} {}", insn1.code.to_string(), insn1.to_string());
        println!("{:20} {}", insn2.code.to_string(), insn2.to_string());
        println!("{:20} {}", insn3.code.to_string(), insn3.to_string());
        println!("{:20} {}", insn4.code.to_string(), insn4.to_string());

        let insn5 = Instruction::from_string(&X86_ARCH, "or ax, bx");
        let insn6 = Instruction::from_string(&X86_ARCH, "or [ax], bx");
        let insn7 = Instruction::from_string(&X86_ARCH, "or ax, [bx]");
        let insn8 = Instruction::from_string(&X86_ARCH, "or ax, 0x1ffff"); 
        println!("{:20} {}", insn5.code.to_string(), insn5.to_string());
        println!("{:20} {}", insn6.code.to_string(), insn6.to_string());
        println!("{:20} {}", insn7.code.to_string(), insn7.to_string());
        println!("{:20} {}", insn8.code.to_string(), insn8.to_string());
        
        let insn9 = Instruction::from_string(&X86_ARCH, "adc ax, bx");
        let insn10 = Instruction::from_string(&X86_ARCH, "adc [ax], bx");
        let insn11 = Instruction::from_string(&X86_ARCH, "adc ax, [bx]");
        let insn12 = Instruction::from_string(&X86_ARCH, "adc ax, 0x1ffff"); 
        println!("{:20} {}", insn9.code.to_string(), insn9.to_string());
        println!("{:20} {}", insn10.code.to_string(), insn10.to_string());
        println!("{:20} {}", insn11.code.to_string(), insn11.to_string());
        println!("{:20} {}", insn12.code.to_string(), insn12.to_string());

        let insn13 = Instruction::from_string(&X86_ARCH, "sbb ax, bx");
        let insn14 = Instruction::from_string(&X86_ARCH, "sbb [ax], bx");
        let insn15 = Instruction::from_string(&X86_ARCH, "sbb ax, [bx]");
        let insn16 = Instruction::from_string(&X86_ARCH, "sbb ax, 0x1ffff"); 
        println!("{:20} {}", insn13.code.to_string(), insn13.to_string());
        println!("{:20} {}", insn14.code.to_string(), insn14.to_string());
        println!("{:20} {}", insn15.code.to_string(), insn15.to_string());
        println!("{:20} {}", insn16.code.to_string(), insn16.to_string());

        let insn17 = Instruction::from_string(&X86_ARCH, "and ax, bx");
        let insn18 = Instruction::from_string(&X86_ARCH, "and [ax], bx");
        let insn19 = Instruction::from_string(&X86_ARCH, "and ax, [bx]");
        let insn20 = Instruction::from_string(&X86_ARCH, "and ax, 0x1ffff"); 
        println!("{:20} {}", insn17.code.to_string(), insn17.to_string());
        println!("{:20} {}", insn18.code.to_string(), insn18.to_string());
        println!("{:20} {}", insn19.code.to_string(), insn19.to_string());
        println!("{:20} {}", insn20.code.to_string(), insn20.to_string());

        let insn21 = Instruction::from_string(&X86_ARCH, "sub ax, bx");
        let insn22 = Instruction::from_string(&X86_ARCH, "sub [ax], bx");
        let insn23 = Instruction::from_string(&X86_ARCH, "sub ax, [bx]");
        let insn24 = Instruction::from_string(&X86_ARCH, "sub ax, 0x1ffff"); 
        println!("{:20} {}", insn21.code.to_string(), insn21.to_string());
        println!("{:20} {}", insn22.code.to_string(), insn22.to_string());
        println!("{:20} {}", insn23.code.to_string(), insn23.to_string());
        println!("{:20} {}", insn24.code.to_string(), insn24.to_string());

        let insn25 = Instruction::from_string(&X86_ARCH, "xor ax, bx");
        let insn26 = Instruction::from_string(&X86_ARCH, "xor [ax], bx");
        let insn27 = Instruction::from_string(&X86_ARCH, "xor ax, [bx]");
        let insn28 = Instruction::from_string(&X86_ARCH, "xor ax, 0x1ffff"); 
        println!("{:20} {}", insn25.code.to_string(), insn25.to_string());
        println!("{:20} {}", insn26.code.to_string(), insn26.to_string());
        println!("{:20} {}", insn27.code.to_string(), insn27.to_string());
        println!("{:20} {}", insn28.code.to_string(), insn28.to_string());

        let insn29 = Instruction::from_string(&X86_ARCH, "cmp ax, bx");
        let insn30 = Instruction::from_string(&X86_ARCH, "cmp [ax], bx");
        let insn31 = Instruction::from_string(&X86_ARCH, "cmp ax, [bx]");
        let insn32 = Instruction::from_string(&X86_ARCH, "cmp ax, 0x1ffff"); 
        println!("{:20} {}", insn29.code.to_string(), insn29.to_string());
        println!("{:20} {}", insn30.code.to_string(), insn30.to_string());
        println!("{:20} {}", insn31.code.to_string(), insn31.to_string());
        println!("{:20} {}", insn32.code.to_string(), insn32.to_string());

        let insn33 = Instruction::from_string(&X86_ARCH, "inc al");
        let insn34 = Instruction::from_string(&X86_ARCH, "inc ebx");
        let insn35 = Instruction::from_string(&X86_ARCH, "dec edx");
        let insn36 = Instruction::from_string(&X86_ARCH, "dec ecx");
        let insn37 = Instruction::from_string(&X86_ARCH, "push edx");
        let insn38 = Instruction::from_string(&X86_ARCH, "pop ecx");
        println!("{:20} {}", insn33.code.to_string(), insn33.to_string());
        println!("{:20} {}", insn34.code.to_string(), insn34.to_string());
        println!("{:20} {}", insn35.code.to_string(), insn35.to_string());
        println!("{:20} {}", insn36.code.to_string(), insn36.to_string());
        println!("{:20} {}", insn37.code.to_string(), insn37.to_string());
        println!("{:20} {}", insn38.code.to_string(), insn38.to_string());
    }

}