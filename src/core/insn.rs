


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::fmt::{self};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::arch::evo::def::{evo_decode, evo_encode, EVO_ARCH};
use crate::arch::info::Arch;
use crate::arch::riscv::def::{riscv32_decode, riscv32_encode, RISCV32_ARCH};
use crate::arch::x86::def::{x86_decode, x86_encode, X86_ARCH};
use crate::{log_error, log_warning};
use crate::util::log::Span;
use crate::core::op::{Opcode, Operand, OPR_IMM};
use crate::core::val::Value;


// ============================================================================== //
//                                 Const
// ============================================================================== //

/// no condition
pub const COND_NO: u16 = 0b0000_0000;
/// x == y
pub const COND_EQ: u16 = 0b0010_0000;
/// x != y
pub const COND_NE: u16 = 0b0011_0000;
/// x < y
pub const COND_LT: u16 = 0b0100_0000;
/// x >= y
pub const COND_GE: u16 = 0b0101_0000;
/// x <= y
pub const COND_LE: u16 = 0b0110_0000;
/// x > y
pub const COND_GT: u16 = 0b0111_0000;
/// is branch insn
pub const INSN_BR: u16 = 0b0000_0001_0000_0000;
/// is jump insn
pub const INSN_JP: u16 = 0b0000_0010_0000_0000;
/// is exit insn
pub const INSN_ET: u16 = 0b0000_0100_0000_0000;
/// have signed operands
pub const INSN_SIG: u16 = 0b0000_0000_0000_0000;
/// all unsigned operands
pub const INSN_USD: u16 = 0b0001_0000_0000_0000;


// ============================================================================== //
//                                insn::RegFile
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegFile(Rc<RefCell<Vec<Rc<RefCell<Operand>>>>>);

impl RegFile {


    // Init Opcode POOL
    thread_local! {
        /// Register pool (Shared Global)
        pub static REG_POOL : Rc<RefCell<HashMap<&'static Arch, RegFile>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    pub fn new(arch: &'static Arch) -> RegFile {
        // 1. check arch is in
        if !Self::reg_pool_is_in(arch) {
            Self::reg_pool_init(arch);
        }
        // 2. get reg
        Self::reg_pool_get(arch)
    }

    /// Define Register Temp and add to pool
    /// ### Register flag
    /// ```txt
    /// (0-7):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ ├─┴─┘   (0-2) 000 is 8-bit , 001 is 16-bit, 010 is 32-bit , 011 is 64-bit
    /// │ │ │ │ │ └────── (0-2) 100 is 80-bit, 101 is 96-bit, 110 is 128-bit, 111 is 256-bit
    /// │ │ │ │ └──────── (3) 0 is little-endian, 1 is big-endian
    /// │ │ │ └────────── <Reserved>
    /// │ │ └──────────── <Reserved>
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// (8-15): offset
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ ├─┴─┴─┘   (8-11) 64-bit scales: 0,   8,   16,  24,  32,  40,  48,  54,  64,  72,  80,  88,  96, 104, 112
    /// │ │ │ │ └──────── (8-11) offset symbol: 000, 001, 010, 011, 100, 101, 110, 111
    /// │ │ │ └────────── <Reserved>
    /// │ │ └──────────── <Reserved>
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// ```
    pub fn def(arch: &'static Arch, name: &'static str, val: Value, flag: u16) -> Rc<RefCell<Operand>> {
        // 1. check arch is in
        if !Self::reg_pool_is_in(arch) {
            Self::reg_pool_init(arch);
        }
        // 2. set reg: if not in, push
        let reg = Operand::reg(name, val, flag);
        if !Self::reg_poolr_is_in(arch, name) {
            Self::reg_poolr_push(arch, reg);
        } else {
            // else set
            Self::reg_poolr_nset(arch, name, reg);
        }
        Self::reg_poolr_nget(arch, name)
    }

    pub fn num(&self) -> usize {
        self.0.borrow().len()
    }

    pub fn idx(&self, name: &str) -> usize {
        for i in 0..self.0.borrow().len() {
            if self.0.borrow()[i].borrow().name() == name {
                return i;
            }
        }
        log_error!("RegFile no such reg: {}", name);
        0
    }

    pub fn val(&self, name: &str) -> Value {
        self.0.borrow()[self.idx(name)].borrow().val()
    }

    pub fn is_in(&self, name: &str) -> bool {
        for i in 0..self.0.borrow().len() {
            if self.0.borrow()[i].borrow().name() == name {
                return true;
            }
        }
        false
    }
    
    pub fn get(&self, index: usize) -> Rc<RefCell<Operand>> {
        if index >= self.0.borrow().len() {
            log_error!("RegFile index out of range: {}", index);
            return Rc::new(RefCell::new(Operand::reg_def()));
        }
        self.0.borrow()[index].clone()
    }

    pub fn nget(&self, name: &str) -> Rc<RefCell<Operand>> {
        self.get(self.idx(name))
    }

    pub fn push(&self, reg: Operand) {
        self.0.borrow_mut().push(Rc::new(RefCell::new(reg)));
    }

    pub fn set(&self, index: usize, reg: Operand) {
        if index >= self.0.borrow().len() {
            log_error!("RegFile index out of range: {}", index);
            return;
        }
        self.0.borrow_mut()[index] = Rc::new(RefCell::new(reg));
    }

    pub fn nset(&self, name: &str, reg: Operand) {
        self.set(self.idx(name), reg);
    }

    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Regs(Nums={}):\n", self.0.borrow().len()));
        for i in 0..self.0.borrow().len() {
            info.push_str(&format!("- {}", &self.0.borrow()[i].borrow().info()));
            info.push('\n');
        }
        info
    }

    pub fn clr(&self) {
        self.0.borrow_mut().clear();
    }


    pub fn reg_pool_set(arch: &'static Arch, rfile: RegFile) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow_mut().insert(arch, rfile));
    }

    pub fn reg_pool_get(arch: &'static Arch) -> RegFile {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().clone())
    }

    pub fn reg_poolr_nget_all(name: &str) -> Rc<RefCell<Operand>> {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().values().find(|rfile| rfile.is_in(name)).unwrap().nget(name))
    }

    pub fn reg_poolr_is_in_all(name: &str) -> bool {
        // find reg in all arch
        Self::REG_POOL.with(|pool_map| pool_map.borrow().values().find(|rfile| rfile.is_in(name)).is_some())
    }

    pub fn reg_pool_del(arch: &'static Arch) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow_mut().remove(arch));
    }

    pub fn reg_pool_info(arch: &'static Arch) -> String {
        Self::REG_POOL.with(|pool_map| {
            let mut info = String::new();
            info.push_str(&format!("Arch: {} ", arch));
            info.push_str(&pool_map.borrow().get(arch).unwrap().info());
            info
        })
    }

    pub fn reg_pool_clr_all() {
        Self::REG_POOL.with(|pool_map| pool_map.borrow_mut().clear());
    }

    pub fn reg_pool_init(arch: &'static Arch) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow_mut().insert(arch, RegFile(Rc::new(RefCell::new(Vec::new())))));
    }

    pub fn reg_pool_is_in(arch: &'static Arch) -> bool {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).is_some())
    }

    pub fn reg_pool_clr(arch: &'static Arch) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().clr());
    }

    pub fn reg_pool_num(arch: &'static Arch) -> usize {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().num())
    }

    pub fn reg_poolr_is_in(arch: &'static Arch, name: &str) -> bool {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().is_in(name))
    }

    pub fn reg_poolr_get(arch: &'static Arch, index: usize) -> Rc<RefCell<Operand>> {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().get(index))
    }

    pub fn reg_poolr_set(arch: &'static Arch, index: usize, value: Operand) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().set(index, value));
    }

    pub fn reg_poolr_idx(arch: &'static Arch, name: &str) -> usize {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().idx(name))
    }

    pub fn reg_poolr_nget(arch: &'static Arch, name: &str) -> Rc<RefCell<Operand>> {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().nget(name))
    }

    pub fn reg_poolr_nset(arch: &'static Arch, name: &str, reg: Operand) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().nset(name, reg));
    }

    pub fn reg_poolr_push(arch: &'static Arch, reg: Operand) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().push(reg));
    }


}



// ============================================================================== //
//                                insn::Instruction
// ============================================================================== //

/// `Instruction`: IR instruction
/// Apply Operand to Opcode, get new Instruction
/// 
/// ## Byte Code
/// - Refence: RISC-V ISA Spec.
/// - Fill bytes according following format:
/// 
/// ### Insn Format Constant Length
/// 
/// ```txt
/// (32-bits):
///  ┌────────┬─────┬─────┬────┬──────┬────┬─────┬────────────────┐
///  │31    25│24 20│19 15│  12│11   7│6  0│ Typ │      Arch      │
///  ├────────┼─────┼─────┼────┼──────┼────┼─────┼────────────────┤
///  │   f7   │ rs2 │ rs1 │ f3 │  rd  │ op │  R  │ RISC-V, EVO    │
///  ├────────┴─────┼─────┼────┼──────┼────┼─────┼────────────────┤
///  │    imm[12]   │ rs1 │ f3 │  rd  │ op │  I  │ RISC-V, EVO    │
///  ├────────┬─────┼─────┼────┼──────┼────┼─────┼────────────────┤
///  │  imm1  │ rs2 │ rs1 │ f3 │ imm0 │ op │  S  │ RISC-V         │
///  ├─┬──────┼─────┼─────┼────┼────┬─┼────┼─────┼────────────────┤
///  │3│  i1  │ rs2 │ rs1 │ f3 │ i0 │2│ op │  B  │ RISC-V         │
///  ├─┴──────┴─────┴─────┴────┼────┴─┼────┼─────┼────────────────┤
///  │          imm[20]        │  rd  │ op │  U  │ RISC-V         │
///  ├─┬──────────┬─┬──────────┼──────┼────┼─────┼────────────────┤
///  │3│   imm0   │1│   imm2   │  rd  │ op │  J  │ RISC-V         │
///  ├─┴──────────┴─┴──────────┴──────┴────┼─────┼────────────────┤
///  │   ?????                             │  ?  │ ?????          │
///  └─────────────────────────────────────┴─────┴────────────────┘
/// ```
/// 
/// ### Insn Format Variable Length
/// ```txt
/// (Variable-length/Byte):
///  ┌──────────────────┬─────────────────────────────────────────┐
///  │     Type: X      │          Arch: i386, x86_64             │
///  ├──────┬───────────┼─────┬──────────────┬────────────┬───┬───┤
///  │ Pref │    Rex    │ Opc │    ModR/M    │    SIB     │ D │ I │
///  ├──────┼───────────┼─────┼──────────────┼────────────┼───┼───┤
///  │ 0,1  │    0,1    │ 1~3 │     0,1      │    0,1     │ 0 │ 0 │
///  ├──────┼───────────┼─────┼──────────────┼────────────┤ 1 │ 1 │
///  │ insn │ 0100 1101 │ ??? │  ?? ??? ???  │ ?? ??? ??? │ 2 │ 2 │
///  │ addr │ ──── ──── │ ─── │  ── ─── ───  │ ── ─── ─── │ 4 │ 4 │
///  │ .... │ patt WRXB │ po  │ mod r/op r/m │ ss idx bas │   │ 8'│
///  └──────┴───────────┴─────┴──────────────┴────────────┴───┴───┘
///  Opcode:
///    0. po: 1~3 Byte of Opcode such as: ADD eax, i32 (po=0x05).
///    1. trans2: 2 Byte po, if first Byte is 0x0f.
///    2. trans3: 3 Byte po, if fisrt and second Bytes are 0x0f 38.
///    3. field extention: po need to concat ModR/M.op(3-bits) field.
///  Imm(I):
///    0. imm: (0,1,2,4,8) Byte of Imm such as: ADD eax 0X4351FF23 (imm=0x23 ff 51 43).
///  Hidden Reg:
///    0. eax: when po=0x05, auto use eax as target reg. (Insn=0x05 23 ff 51 43)
///  ModR/M:
///    0~2. r/m - As Direct/Indirect reg operand(E).
///    3~5. r/op - As Reg ref(G), or as 3-bit opcode extension.
///    6~7. mod - 0b11: Reg-Direct Addressing mode(Reg), else Reg-Indirect Addressing mode(Mem).
///    Such as: ADD ecx, esi (po=0x01), set ModR/M: 0b11 110(esi) 001(ecx)=0xf1.
///    Get (Insn=0x01 f1)
///  Prefix(Legacy):
///    - instruction prefix
///    - address-size prefix
///    - operand-size prefix: 0x66(32 -> 16)
///    - segment override prefix
///    Such as: MOV r/m32, r32 (po=0x89), set opr-prefix: 0x66
///    Get MOV r/m16, r16 (Insn=0x66 89 ..)
///  Rex Prefix(x86_64):
///    0. B - Extension of SIB.base field.
///    1. X - Extension of SIB.idx field.
///    2. R - Extension of ModR/M.reg field (Reg Num: 8 -> 16).
///    3. W - 0: 64-bits operand, 1: default(32-bits) operand.
///    5~8. 0100 - Fixed bit patten.
///    Such as: ADD rcx, rsi (po=0x01, 64-bits), set Rex: 0b0100 1000=0x48.
///    Get (Insn=0x48 01 f1)
///    Such as: ADD rcx, r9(0b1001) (po=0x01, 64-bits), set Rex: 0b0100 1100=0x4c.
///    Set ModR/M: 0b11 001 001=0xc9, Get (Insn=0x4c 01 c9)
///  Disp(D):
///    0. imm: (0,1,2,4) Byte of Imm as addr disp.
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// ### Instruction flag
    /// ```txt
    /// (0-7):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ ├─┴─┘   (0-2) 000 is 8-bit , 001 is 16-bit, 010 is 32-bit , 011 is 64-bit
    /// │ │ │ │ │ └────── (0-2) 100 is 80-bit, 101 is 96-bit, 110 is 128-bit, 111 is 256-bit
    /// │ │ │ │ └──────── (3) 0 is little-endian, 1 is big-endian
    /// │ │ │ │         ┌ (4-7) 0000: COND_NO, 0001: <Reserved>
    /// │ │ │ │         │ (4-7) 0010: COND_EQ, 0011: COND_NE
    /// ├─┴─┴─┘         │ (4-7) 0100: COND_LT, 0101: COND_GE
    /// └───────────────┴ (4-7) 0110: COND_LE, 0111: COND_GT
    /// (8-15):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ │ │ └── (8) 0: is not branch, 1: is branch
    /// │ │ │ │ │ │ └──── (9) 0: is not jump, 1: is jump
    /// │ │ │ │ │ └────── (10) 0: is not exit, 1: is exit
    /// │ │ │ │ └──────── <Reserved>
    /// │ │ │ └────────── (12) 0: have signed operands, 1: all unsigned operands
    /// │ │ └──────────── <Reserved>
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// ```
    pub flag: u16,
    pub opc : Opcode,
    pub opr : Vec<Operand>,
    pub opb : &'static str,
    pub code : Value,
    pub arch : &'static Arch,
    pub is_applied : bool,
    /// encode func
    pub enc : Option<fn(&mut Instruction, Vec<Operand>) -> Instruction>,
}

impl Instruction {

    // Init Opcode POOL
    thread_local! {
        /// Register pool (Shared Global)
        pub static REG_POOL : Rc<RefCell<Vec<Rc<RefCell<Operand>>>>> = Rc::new(RefCell::new(Vec::new()));
        /// IR Insn pool (Shared Global)
        pub static INSN_POOL: Rc<RefCell<Vec<Rc<RefCell<Instruction>>>>> = Rc::new(RefCell::new(Vec::new()));
    }

    /// Get Undef Insn
    pub fn undef() -> Instruction {
        Instruction {
            flag: 0,
            opc: Opcode::new(".insn", Vec::new(), "Undef"),
            opr: Vec::new(),
            opb: "",
            code: Value::u32(0),
            arch: &EVO_ARCH,
            is_applied: false,
            enc: None,
        }
    }

    /// Define Instruction Temp and add to pool
    pub fn def(arch: &'static Arch, name: &'static str, flag: u16, syms: Vec<u16>, ty: &'static str, opb: &'static str) -> Instruction {
        let opc = Opcode::new(name, syms, ty);
        let insn = Instruction {
            flag: flag,
            opc: opc.clone(),
            opr: Vec::new(),
            opb,
            code: Value::from_string(opb),
            arch,
            is_applied: false,
            enc: Self::encode_pool_init(arch)
        };
        // add to pool
        Instruction::insn_pool_push(insn);
        // return
        Instruction::insn_pool_nget(name).borrow().clone()
    }

    /// Encode
    pub fn encode(&mut self, opr: Vec<Operand>) -> Instruction {
        if self.enc.is_some() {
            self.enc.unwrap()(self, opr)
        } else {
            log_warning!("Encoder not found arch: {}", self.arch);
            self.clone()
        }
    }

    /// Decode: you should init pool first
    pub fn decode(arch: &'static Arch, value: Value) -> Instruction {
        Self::decode_pool_init(arch).unwrap()(value)
    }

    /// apply temp to Instruction
    pub fn apply(name: &'static str, opr: Vec<Operand>) -> Instruction {
        let mut insn = Instruction::insn_pool_nget(name).borrow().clone();
        insn.encode(opr)
    }

    /// route of encode pool
    pub fn encode_pool_init(arch: &'static Arch) -> Option<fn(&mut Instruction, Vec<Operand>) -> Instruction> {
        match *arch {
            EVO_ARCH => Some(evo_encode),
            X86_ARCH => Some(x86_encode),
            RISCV32_ARCH => Some(riscv32_encode),
            _ => {
                log_warning!("Encoder init fail, not support arch: {}", arch);
                None
            }
        }
    }

    /// route of decode pool
    pub fn decode_pool_init(arch: &'static Arch) -> Option<fn(value: Value) -> Instruction> {
        match *arch {
            EVO_ARCH => Some(evo_decode),
            X86_ARCH => Some(x86_decode),
            RISCV32_ARCH => Some(riscv32_decode),
            _ => {
                log_warning!("Decoder init fail, not support arch: {}", arch);
                None
            }
        }
    }

    // ================= Instruction.insn_pool ================== //

    /// delete Insn from pool
    pub fn insn_pool_del(name: &'static str) {
        // Get index
        let idx = Self::insn_pool_idx(name);
        // Delete
        Self::INSN_POOL.with(|pool| pool.borrow_mut().remove(idx));
    }

    /// Get Insn index from pool
    pub fn insn_pool_idx(name: &'static str) -> usize {
        Self::INSN_POOL.with(|pool| pool.borrow().iter().position(|r| r.borrow().name() == name).unwrap())
    }

    /// get Insn from pool
    pub fn insn_pool_nget(name: &'static str) -> Rc<RefCell<Instruction>> {
        Self::INSN_POOL.with(|pool| pool.borrow().iter().find(|r| r.borrow().name() == name).unwrap().clone())
    }

    /// Set Insn from pool
    pub fn insn_pool_nset(name: &'static str, insn: Instruction) {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().iter_mut().find(|r| r.borrow().name() == name).unwrap().replace(insn));
    }

    /// Insert Insn from pool
    pub fn insn_pool_push(insn: Instruction) {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(insn))));
    }

    /// Get Insn from pool
    pub fn insn_pool_get(index: usize) -> Rc<RefCell<Instruction>> {
        Self::INSN_POOL.with(|pool| pool.borrow()[index].clone())
    }

    /// Set Insn from pool
    pub fn insn_pool_set(index: usize, insn: Instruction) {
        Self::INSN_POOL.with(|pool| pool.borrow_mut()[index].replace(insn));
    }

    /// Clear Temp Insn Pool
    pub fn insn_pool_clr() {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Get info of Insn Pool
    pub fn insn_pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Instructions (Num = {}):\n", Self::insn_pool_size()));
        
        for i in 0..Self::insn_pool_size() {
            let insn = Self::insn_pool_get(i).borrow().clone();
            info.push_str(&format!("- {:<40}   {} ({})\n", insn.info(), insn.ty(), insn.opb));
        }
        info
    }

    /// Get size of Insn Pool
    pub fn insn_pool_size() -> usize {
        Self::INSN_POOL.with(|pool| pool.borrow().len())
    }

    // ================== Instruction.reg_pool ================== //

    // /// Pool set reg by index
    // pub fn reg_pool_set(idx: usize, reg: Operand) {
    //     Self::REG_POOL.with(|pool| pool.borrow_mut()[idx] = Rc::new(RefCell::new(reg)));
    // }

    // /// Pool push reg
    // pub fn reg_pool_push(reg: Operand) {
    //     Self::REG_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(reg))));
    // }

    // /// Pool get reg refenence by index
    // pub fn reg_pool_get(idx: usize) -> Rc<RefCell<Operand>> {
    //     Self::REG_POOL.with(|pool| pool.borrow()[idx].clone())
    // }

    // /// Pool set reg `index` value by index
    // pub fn reg_pool_set_val(idx: usize, val: Value) { 
    //     Self::REG_POOL.with(|pool| pool.borrow_mut()[idx].borrow_mut().set_reg(val));
    // }

    // /// Pool get reg `index` value by index
    // pub fn reg_pool_get_val(idx: usize) -> Value {
    //     Self::REG_POOL.with(|pool| pool.borrow()[idx].borrow().val())
    // }

    // /// Pool set reg by name
    // pub fn reg_pool_nset(name: &str , reg: Operand) {
    //     Self::REG_POOL.with(|pool| pool.borrow_mut().iter_mut().find(|r| r.borrow().name() == name).unwrap().replace(reg));
    // }

    // /// Pool get reg refenence by name
    // pub fn reg_pool_nget(name: &str) -> Rc<RefCell<Operand>> {
    //     Self::REG_POOL.with(|pool| pool.borrow().iter().find(|r| r.borrow().name() == name).unwrap().clone())
    // }

    // /// Pool check reg is in
    // pub fn reg_pool_is_in(name: &str) -> bool {
    //     Self::REG_POOL.with(|pool| pool.borrow().iter().any(|r| r.borrow().name() == name))
    // }

    // /// Pool get reg pool idx by name
    // pub fn reg_pool_idx(name: &str) -> usize {
    //     Self::REG_POOL.with(|pool| pool.borrow().iter().position(|r| r.borrow().name() == name).unwrap())
    // }

    // /// Pool clear
    // pub fn reg_pool_clr() {
    //     Self::REG_POOL.with(|pool| pool.borrow_mut().clear());
    // }

    // /// Pool size
    // pub fn reg_pool_size() -> usize {
    //     Self::REG_POOL.with(|pool| pool.borrow().len())
    // }

    // /// Pool info
    // pub fn reg_pool_info() -> String {
    //     let mut info = String::new();
    //     info.push_str(&format!("Registers (Num = {}):\n", Self::reg_pool_size()));
    //     for i in 0..Self::reg_pool_size() {
    //         let reg = Self::reg_pool_get(i).borrow().clone();
    //         info.push_str(&format!("- {}\n", reg.info()));
    //     }
    //     info
    // }


    // ==================== Instruction.get ===================== //
    
    /// Get byte size of Instruction
    pub fn size(&self) -> usize {
        self.code.size()
    }

    /// Name of Instruction
    pub fn name(&self) -> &'static str {
        self.opc.name()
    }

    /// Syms of Instruction
    pub fn syms(&self) -> Vec<u16> {
        self.opc.syms()
    }

    /// Show binary byte string of Instruction
    pub fn bin(&self, index: usize, byte_num: i32, big_endian: bool) -> String {
        self.code.bin(index, byte_num, big_endian)
    }

    /// Show type of Instruction
    pub fn ty(&self) -> &'static str {
        self.opc.ty()
    }

    /// Get String of Opcode + Operand Syms
    pub fn info(&self) -> String {
        format!("{}", self.opc.to_string())
    }

    /// Get binary define str of insn
    pub fn opdef(&self) -> &'static str {
        self.opb
    }


    // ==================== Instruction.set ===================== //


    /// flush insn arch
    pub fn set_arch(&mut self, arch: &'static Arch) {
        self.arch = arch;
        self.enc = Self::encode_pool_init(arch);
    }

    // ==================== Instruction.flag ==================== //

    /// is jump instruction
    pub fn is_jump(&self) -> bool {
        (self.flag & INSN_JP) == INSN_JP
    }

    /// is branch instruction
    pub fn is_branch(&self) -> bool {
        (self.flag & INSN_BR) == INSN_BR
    }
    

    // ==================== Instruction.sym ===================== //

    /// Check Operand Syms
    pub fn check_syms(&self, opr: Vec<Operand>) -> bool {
        self.opc.check_syms(opr.iter().map(|x| x.sym()).collect())
    }

    /// Reset Operand Syms
    pub fn set_syms(&mut self, opr: Vec<Operand>) {
        self.opc.set_syms(opr.iter().map(|x| x.sym()).collect());
    }


    // ==================== Instruction.code ==================== //
    
    /// Get Byte Code clone of Instruction
    pub fn code(&self) -> Value {
        if !self.is_applied {
            log_error!("Code not applied: {} ", self.opc.name());
        }
        self.code.clone()
    }

    /// Tail add code
    pub fn code_append(&mut self, code: Value) {
        self.code.append(code);
    }

    /// Head add code
    pub fn code_insert(&mut self, code: Value) {
        self.code.insert(code);
    }

    // ========================================================== //
    //                         riscv32
    // ========================================================== //

    pub fn funct7(&self) -> u8 {
        self.code.get_ubyte(0, 25, 7)
    }

    pub fn funct3(&self) -> u8 {
        self.code.get_ubyte(0, 12, 3)
    }

    pub fn opcode(&self) -> u8 {
        self.code.get_ubyte(0, 0, 6)
    }

    pub fn rs1(&self) -> u8 {
        self.code.get_ubyte(0, 15, 5)
    }

    pub fn rs2(&self) -> u8 {
        self.code.get_ubyte(0, 20, 5)
    }

    pub fn rd(&self) -> u8 {
        self.code.get_ubyte(0, 7, 5)
    }

    pub fn imm_i(&self) -> u16 {
        self.code.get_uhalf(0, 20, 12)
    }

    pub fn imm_s(&self) -> u16 {
        let imm0 = self.code.get_ubyte(0, 7, 5);
        let imm1 = self.code.get_ubyte(0, 25, 7);
        let res = (imm0 as u16) | ((imm1 as u16) << 5);
        res
    }

    pub fn imm_b(&self) -> u16 {
        let imm0 = self.code.get_ubyte(0, 8, 4);
        let imm1 = self.code.get_ubyte(0, 25, 6);
        let imm2 = self.code.get_ubyte(0, 7, 1);
        let imm3 = self.code.get_ubyte(0, 31, 1);
        let res = (imm0 as u16) | ((imm1 as u16) << 4) | ((imm2 as u16) << 11) | ((imm3 as u16) << 12);
        res
    }

    pub fn imm_u(&self) -> u32 {
        self.code.get_uword(0, 12, 20)
    }

    pub fn imm_j(&self) -> u32 {
        let imm0 = self.code.get_uhalf(0, 21, 10);
        let imm1 = self.code.get_ubyte(0, 20, 1);
        let imm2 = self.code.get_ubyte(0, 12, 8);
        let imm3 = self.code.get_ubyte(0, 31, 1);
        let res = (imm0 as u32) | ((imm1 as u32) << 11) | ((imm2 as u32) << 19) | ((imm3 as u32) << 20);
        res
    }

    pub fn set_rs1(&mut self, rs1: u8) {
        self.code.set_ubyte(0, 15, 5, rs1);
    }

    pub fn set_rs2(&mut self, rs2: u8) {
        self.code.set_ubyte(0, 20, 5, rs2);
    }

    pub fn set_rd(&mut self, rd: u8) {
        self.code.set_ubyte(0, 7, 5, rd);
    }

    pub fn set_imm_i(&mut self, imm: u16) {
        self.code.set_uhalf(0, 20, 12, imm);
    }

    pub fn set_imm_s(&mut self, imm: u16) {
        let imm0 = (imm & 0x1f) as u8;
        let imm1 = (imm >> 5) as u8;
        self.code.set_ubyte(0, 7, 5, imm0);
        self.code.set_ubyte(0, 25, 7, imm1);
    }

    pub fn set_imm_b(&mut self, imm: u16) {
        let imm0 = (imm & 0xf) as u8;
        let imm1 = ((imm >> 5) & 0x3f) as u8;
        let imm2 = ((imm >> 11) & 0x1) as u8;
        let imm3 = ((imm >> 12) & 0x1) as u8;
        self.code.set_ubyte(0, 8, 4, imm0);
        self.code.set_ubyte(0, 25, 6, imm1);
        self.code.set_ubyte(0, 7, 1, imm2);
        self.code.set_ubyte(0, 31, 1, imm3);
    }

    pub fn set_imm_u(&mut self, imm: u32) {
        self.code.set_uword(0, 12, 20, imm);
    }

    pub fn set_imm_j(&mut self, imm: u32) {
        let imm0 = (imm & 0x3ff) as u16;
        let imm1 = ((imm >> 11) & 0x1) as u8;
        let imm2 = ((imm >> 19) & 0xff) as u8;
        let imm3 = ((imm >> 20) & 0x1) as u8;
        self.code.set_uhalf(0, 21, 10, imm0);
        self.code.set_ubyte(0, 20, 1, imm1);
        self.code.set_ubyte(0, 12, 8, imm2);
        self.code.set_ubyte(0, 31, 1, imm3);
    }


    // ========================================================== //
    //                         x86
    // ========================================================== //



    // ==================== Instruction.str ===================== //

    /// To string: `[opc] [opr1] : [sym1], [opr2] : [sym2], ...`
    pub fn to_string(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("{:<10} ", self.opc.name()));
        for i in 0..self.opr.len() {
            let r = self.opr[i].clone();
            info.push_str(&format!("{}", r.to_string()));
            // if iter not last push `,`
            if i < self.opr.len() - 1 {
                info.push_str(&format!(", "));
            }
        }
        info
    }

    /// From string to Instruction
    pub fn from_string(str: &'static str) -> Instruction {
        // 1. Deal with string
        let str = str.trim();
        // Check if the string has space
        if !str.contains(' ') {
            let name = str;
            let mut res = Instruction::insn_pool_nget(name).borrow().clone();
            res = res.encode(vec![]);
            return res;
        }
        // 2. Divide in first space and Get Opcode: `[opc] [opr1], [opr2], ...`
        let mut part = str.splitn(2, ' ');
        // 3. Find Insn from pool
        let res = Instruction::insn_pool_nget(part.next().unwrap().trim());
        // 4. Collect extra part: Divide by `,` and Get Operands str
        let extra_part = part.next().unwrap();
        let opr = extra_part.split(',').collect::<Vec<_>>();
        // 5. Get Operands by sym
        let opr_sym = res.borrow().syms();
        let mut opr_vec = Vec::new();
        for i in 0..opr_sym.len() {
            let mut r = Operand::from_string(opr_sym[i], opr[i].trim());
            if opr_sym[i] == OPR_IMM {
                r = Operand::imm(Value::u32(r.val().get_word(0)));
            }
            opr_vec.push(r);
        }
        // 5. Encode Instruction
        let res = res.borrow_mut().encode(opr_vec);
        res
    }
}


impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}




// ============================================================================== //
//                                insn::BasicBlock
// ============================================================================== //


/// BasicBlock
pub struct BasicBlock {
    
    pub flag: u16,
    pub src_insns: Vec<Instruction>,
    pub ir_insns: Vec<Instruction>,
    pub trg_insns: Vec<Instruction>,
    pub liveness_regs: Vec<Operand>,

    /// If the block is lifted to EVO ir arch: src -> ir
    pub is_lifted: bool,
    /// If the block is lowered to target arch: ir -> trg
    pub is_lowered: bool,
}


impl BasicBlock {


    pub fn init(src_insns: Vec<Instruction>, flag: u16) -> BasicBlock {
        Self {
            flag,
            src_insns,
            ir_insns: vec![],
            trg_insns: vec![],
            liveness_regs: vec![],
            is_lifted: false,
            is_lowered: false
        }
    }

}