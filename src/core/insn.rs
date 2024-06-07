


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::fmt::{self};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::arch::evo::def::{evo_decode, evo_encode, EVO_ARCH};
use crate::arch::info::{Arch, BIT32, BIT64, LITTLE_ENDIAN};
use crate::arch::riscv::def::{riscv32_decode, riscv32_encode, RISCV32_ARCH};
use crate::arch::x86::def::{x86_decode, x86_encode, X86_ARCH};
use crate::{log_error, log_warning};
use crate::util::log::Span;
use crate::core::op::{Opcode, Operand, REG_LOCA};
use crate::core::val::Value;


// ============================================================================== //
//                                 Const
// ============================================================================== //

/// no cond    0  : return false
pub const COND_NO: u8 = 0b0000;
/// always     1  : return true
pub const COND_AL: u8 = 0b0001;
/// x == y     2
pub const COND_EQ: u8 = 0b0010;
/// x != y     3
pub const COND_NE: u8 = 0b0011;
/// x < y      4
pub const COND_LT: u8 = 0b0100;
/// x <= y     5
pub const COND_LE: u8 = 0b0101;
/// x > y      6
pub const COND_GT: u8 = 0b0110;
/// x >= y     7
pub const COND_GE: u8 = 0b0111;

/// Trap: Find next TB start addr 
pub const TRAP_FTB: u8 = 0;
/// Trap: Guest PC => Find undefined insn Op
pub const TRAP_UND: u8 = 1;
/// Trap: Host Insn => Call Syscall Num
pub const TRAP_SYS: u8 = 2;
/// Trap: Guest Addr => Dynamic Jump
pub const TRAP_DYN: u8 = 3;
/// Trap: Guest Mem => Mem access fault
pub const TRAP_MAF: u8 = 4;


pub const INSN_BR_NO: u16 = (COND_NO << 4) as u16;
pub const INSN_BR_AL: u16 = (COND_AL << 4) as u16;
pub const INSN_BR_EQ: u16 = (COND_EQ << 4) as u16;
pub const INSN_BR_NE: u16 = (COND_NE << 4) as u16;
pub const INSN_BR_LT: u16 = (COND_LT << 4) as u16;
pub const INSN_BR_LE: u16 = (COND_LE << 4) as u16;
pub const INSN_BR_GT: u16 = (COND_GT << 4) as u16;
pub const INSN_BR_GE: u16 = (COND_GE << 4) as u16;

/// is branch insn
pub const INSN_BR: u16 = 0b0000_0001_0000_0000;
/// is jump insn
pub const INSN_JP: u16 = 0b0000_0010_0000_0000;
/// is exit insn
pub const INSN_ET: u16 = 0b0000_0100_0000_0000;
/// res is signed operands
pub const INSN_SIG: u16 = 0b0000_0000_0000_0000;
/// res is unsigned operands
pub const INSN_USD: u16 = 0b0001_0000_0000_0000;


// ============================================================================== //
//                                insn::RegFile
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegFile{
    rf: Rc<RefCell<Vec<Rc<RefCell<Operand>>>>>,
    arch: &'static Arch
}

impl RegFile {


    // Init Opcode POOL
    thread_local! {
        /// Register pool (Shared Global)
        pub static REG_POOL : Rc<RefCell<HashMap<&'static Arch, RegFile>>> = Rc::new(RefCell::new(HashMap::new()));
        /// Register Map (Shared Global)
        pub static REG_MAP_POOL: Rc<RefCell<HashMap<(&'static Arch, &'static Arch, usize), Rc<RefCell<Operand>>>>> = Rc::new(RefCell::new(HashMap::new()));
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
    /// │ │ ├─┘
    /// │ │ └──────────── Register Type: 00 Global, 01 Bundled, 10 Temp, 11 Local
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// (8-15): offset
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ ├─┴─┴─┘   (8-11) 64-bit scales: 0,   8,   16,  24,  32,  40,  48,  54,  64,  72,  80,  88,  96, 104, 112
    /// │ │ │ │ └──────── (8-11) offset symbol: 0000,0001,0010,0011,0100,0101,0110,0111,1000,1001,1010,1011,1100,1101,1110
    /// │ │ │ └────────── <Reserved>
    /// │ │ └──────────── <Reserved>
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// ```
    pub fn def(arch: &'static Arch, name: &'static str, idx: Value, flag: u16) -> Rc<RefCell<Operand>> {
        // 1. check arch is in
        if !Self::reg_pool_is_in(arch) {
            Self::reg_pool_init(arch);
        }
        // 2. set reg: if not in, push
        let reg = Operand::reg(name, idx, flag);
        if !Self::reg_poolr_is_in(arch, name) {
            Self::reg_poolr_push(arch, reg);
        } else {
            // else set
            Self::reg_poolr_nset(arch, name, reg);
        }
        Self::reg_poolr_nget(arch, name)
    }

    /// Number of registers
    pub fn num(&self) -> usize {
        self.rf.borrow().len()
    }

    /// Max number of registers
    pub fn contain(&self) -> usize {
        self.arch.reg_num
    }

    pub fn last(&self) -> Rc<RefCell<Operand>> {
        self.rf.borrow()[self.rf.borrow().len() - 1].clone()
    }

    pub fn idx(&self, name: &str) -> usize {
        for i in 0..self.rf.borrow().len() {
            if self.rf.borrow()[i].borrow().name() == name {
                return i;
            }
        }
        log_error!("RegFile no such reg: {}", name);
        0
    }

    pub fn val(&self, name: &str) -> Value {
        self.rf.borrow()[self.idx(name)].borrow().val()
    }

    pub fn is_in(&self, name: &str) -> bool {
        for i in 0..self.rf.borrow().len() {
            if self.rf.borrow()[i].borrow().name() == name {
                return true;
            }
        }
        false
    }
    
    pub fn get(&self, index: usize) -> Rc<RefCell<Operand>> {
        if index >= self.rf.borrow().len() {
            log_error!("RegFile index out of range: {}", index);
            return Rc::new(RefCell::new(Operand::reg_def()));
        }
        self.rf.borrow()[index].clone()
    }

    /// Get first global reg in RegFile
    pub fn get_global(&self) -> (usize, Rc<RefCell<Operand>>) {
        for i in 0..self.rf.borrow().len() {
            if self.rf.borrow()[i].borrow().is_reg_global() {
                return (i, self.rf.borrow()[i].clone());
            }
        }
        log_error!("RegFile no global reg!");
        (0, Rc::new(RefCell::new(Operand::reg_def())))
    }

    /// Alloc a local reg in RegFile
    pub fn new_local(&self) -> (usize, Rc<RefCell<Operand>>) {
        if self.num() < self.contain() {
            Self::def(self.arch, "<Loc>", Value::bit(8, self.num() as i128), REG_LOCA | LITTLE_ENDIAN);
            return (self.num(), self.last());
        }
        log_error!("RegFile no extra local reg, contain: {}!", self.contain());
        (0, Rc::new(RefCell::new(Operand::reg_def())))
    }

    /// Free a local reg in RegFile
    pub fn free_local(&self, index: usize) {
        if index >= self.rf.borrow().len() {
            log_error!("RegFile index out of range: {}", index);
            return;
        }
        self.rf.borrow_mut().remove(index);
    }

    pub fn nget(&self, name: &str) -> Rc<RefCell<Operand>> {
        self.get(self.idx(name))
    }

    pub fn push(&self, reg: Operand) {
        self.rf.borrow_mut().push(Rc::new(RefCell::new(reg)));
    }

    pub fn set(&self, index: usize, reg: Operand) {
        if index >= self.rf.borrow().len() {
            log_error!("RegFile index out of range: {}", index);
            return;
        }
        self.rf.borrow_mut()[index] = Rc::new(RefCell::new(reg));
    }

    pub fn nset(&self, name: &str, reg: Operand) {
        self.set(self.idx(name), reg);
    }

    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Regs(Nums={}):\n", self.rf.borrow().len()));
        for i in 0..self.rf.borrow().len() {
            info.push_str(&format!("- {}", &self.rf.borrow()[i].borrow().info()));
            info.push('\n');
        }
        info
    }

    pub fn del(&self, index: usize) {
        self.rf.borrow_mut().remove(index);
    }

    pub fn ndel(&self, name: &str) {
        self.del(self.idx(name));
    }

    pub fn clr(&self) {
        self.rf.borrow_mut().clear();
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
        Self::REG_POOL.with(|pool_map| pool_map.borrow_mut().insert(arch, RegFile{ 
            rf: Rc::new(RefCell::new(Vec::new())),
            arch
        }));
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

    pub fn reg_poolr_last(arch: &'static Arch) -> Rc<RefCell<Operand>> {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().last())
    }

    pub fn reg_poolr_get_global(arch: &'static Arch) -> (usize, Rc<RefCell<Operand>>) {
        // find reg in arch when get a global reg exit
        Self::REG_POOL.with(|pool_map| 
            pool_map.borrow().get(arch).unwrap().get_global()
        )
    }

    pub fn reg_poolr_new_local(arch: &'static Arch) -> (usize, Rc<RefCell<Operand>>) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().new_local())
    }

    pub fn reg_poolr_free_local(arch: &'static Arch, index: usize) {
        Self::REG_POOL.with(|pool_map| pool_map.borrow().get(arch).unwrap().free_local(index));
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


    /// Map reg from src_arch to trg_arch
    pub fn reg_map_pool_set(src_arch: &'static Arch, trg_arch: &'static Arch, src_idx: usize, trg_idx: usize) {
        Self::REG_MAP_POOL.with(|pool_map| {
            let src_reg = Self::reg_poolr_get(src_arch, src_idx);
            let trg_reg = Self::reg_poolr_get(trg_arch, trg_idx);
            src_reg.borrow_mut().set_reg_bund();
            trg_reg.borrow_mut().set_reg_bund();
            pool_map.borrow_mut().insert((src_arch, trg_arch, src_idx), trg_reg.clone());
        });
    }

    pub fn reg_map_pool_get(src_arch: &'static Arch, trg_arch: &'static Arch, src_idx: usize) -> Rc<RefCell<Operand>> {
        Self::REG_MAP_POOL.with(|pool_map| pool_map.borrow().get(&(src_arch, trg_arch, src_idx)).unwrap().clone())
    }

    pub fn reg_map_pool_info(src_arch: &'static Arch, trg_arch: &'static Arch) -> String {
        Self::REG_MAP_POOL.with(|pool_map| {
            let mut info = String::new();
            info.push_str(&format!("Reg Map(Nums={}): {} -> {} \n", Self::reg_map_pool_num(src_arch, trg_arch), src_arch, trg_arch));
            // sreg -> treg
            for (k, v) in pool_map.borrow().iter() {
                if k.0 == src_arch && k.1 == trg_arch {
                    info.push_str(&format!("{} -> {} \n", Self::reg_poolr_get(k.0, k.2).borrow().clone(), v.borrow().clone()));
                }
            }
            info
        })
    }

    pub fn reg_map_pool_nset(src_arch: &'static Arch, trg_arch: &'static Arch, src_name: &str, trg_name: &str) {
        let src_idx =  Self::reg_poolr_nget(src_arch, src_name).borrow().clone().val().get_byte(0) as usize;
        let trg_idx =  Self::reg_poolr_nget(trg_arch, trg_name).borrow().clone().val().get_byte(0) as usize;
        Self::reg_map_pool_set(src_arch, trg_arch, src_idx, trg_idx);
    }

    pub fn reg_map_pool_nget(src_arch: &'static Arch, trg_arch: &'static Arch, src_name: &str) -> Rc<RefCell<Operand>> {
        let src_idx =  Self::reg_poolr_nget(src_arch, src_name).borrow().clone().val().get_byte(0) as usize;
        Self::reg_map_pool_get(src_arch, trg_arch, src_idx)
    }

    pub fn reg_map_pool_is_in(src_arch: &'static Arch, trg_arch: &'static Arch, src_idx: usize) -> bool {
        Self::REG_MAP_POOL.with(|pool_map| pool_map.borrow().contains_key(&(src_arch, trg_arch, src_idx)))
    }

    /// Map reg from src_arch to trg_arch
    pub fn reg_map_pool_num(src_arch: &'static Arch, trg_arch: &'static Arch) -> usize {
        Self::REG_MAP_POOL.with(|pool_map| 
            // if key is src_arch and trg_arch, add 1
            pool_map.borrow().keys().filter(|k| k.0 == src_arch && k.1 == trg_arch).count()
        )
    }

    pub fn reg_map_pool_del(src_arch: &'static Arch, trg_arch: &'static Arch, src_idx: usize) {
        Self::REG_MAP_POOL.with(|pool_map| {
            let src_reg = Self::reg_poolr_get(src_arch, src_idx);
            let trg_reg = Self::reg_map_pool_get(src_arch, trg_arch, src_idx);
            src_reg.borrow_mut().set_reg_global();
            trg_reg.borrow_mut().set_reg_global();
            pool_map.borrow_mut().remove(&(src_arch, trg_arch, src_idx))
        });
    }

    pub fn reg_map_pool_ndel(src_arch: &'static Arch, trg_arch: &'static Arch, src_name: &str) {
        let src_idx =  Self::reg_poolr_nget(src_arch, src_name).borrow().clone().val().get_byte(0) as usize;
        Self::reg_map_pool_del(src_arch, trg_arch, src_idx);
    }

    pub fn reg_map_pool_clr(src_arch: &'static Arch, trg_arch: &'static Arch) {
        Self::REG_MAP_POOL.with(|pool_map| 
            // if key is src_arch and trg_arch, remove
            pool_map.borrow_mut().keys().filter(|k| k.0 == src_arch && k.1 == trg_arch).for_each(|k| {
                pool_map.borrow_mut().remove(k);
            })
        );
    }

    pub fn reg_map_pool_clr_all() {
        Self::REG_MAP_POOL.with(|pool_map| pool_map.borrow_mut().clear());
    }


    pub fn reg_bundle(src_arch: &'static Arch, trg_arch: &'static Arch, src_name: &str, trg_name: &str) {
        Self::reg_map_pool_nset(src_arch, trg_arch, src_name, trg_name);
    }

    pub fn reg_release(src_arch: &'static Arch, trg_arch: &'static Arch, src_name: &str) {
        Self::reg_map_pool_ndel(src_arch, trg_arch, src_name);
    }

    pub fn reg_alloc(src_arch: &'static Arch, trg_arch: &'static Arch, src_idx: usize) -> Rc<RefCell<Operand>> {
        let src_reg = RegFile::reg_poolr_get(src_arch, src_idx);
        if src_reg.borrow().is_reg_bund() {
            RegFile::reg_map_pool_get(src_arch, trg_arch, src_idx)
        } else {
            // find global reg
            let (trg_idx, trg_reg) = RegFile::reg_poolr_get_global(trg_arch);
            // bundle reg
            RegFile::reg_map_pool_set(src_arch, trg_arch, src_idx, trg_idx);
            trg_reg
        }
    }

    pub fn reg_new_local(arch: &'static Arch) -> (usize, Rc<RefCell<Operand>>) {
        RegFile::reg_poolr_new_local(arch)
    }

    pub fn reg_free_local(arch: &'static Arch, idx: usize) {
        RegFile::reg_poolr_free_local(arch, idx)
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
/// - riscv format:
/// ```txt
/// (32-bits):  <-- Big Endian View.
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
/// - evo format:
/// ```txt
///  Max reg nums: 2^8 = 256
///  Max general opcode nums: 2^8 = 256 (Without bits/sign mode)
///  Max extend 1 opcode nums: 2^8 * 2^8 = 65536 (Without bits/sign mode)
///  --> Little Endian View.
///  ┌────────────────┬───────────────────────────────────────────┐
///  │    Type: E     │             Arch: EVO                     │
///  ├────────────────┼────────────────────────┬──────────────────┤
///  │   flag & Op    │        A field         │     B field      │
///  ├────────────┬───┼────────────────────────┼──────────────────┤
///  │     1      │ 1 │          ???           │       ???        │
///  ├────────────┼───┼────────────────────────┼──────────────────┤
///  │ 000 000 00 │ o │ 000.x: off(0)          │ 000.x: off(0)    │
///  │ ─── ─── ── │ p │ 001.x: reg(1)          │ 001.0: imm(4)    │
///  │ BBB AAA sb │ c │ 010.x: reg(1,1)        │ 001.1: imm(8)    │
///  │         ^^ │ o │ 011.x: reg(1,1,1)      │ 010.0: imm(4,4)  │
///  │       flag │ d │ 100.x: reg(1,1,1,1)    │ 010.1: imm(8,8)  │
///  │            │ e │ 101.0: reg(1,1)imm(4)  │ 011.0: imm(4,4,4)│
///  │            │   │ 101.1: reg(1,1)imm(8)  │ 011.1: imm(8,8,8)│
///  │            │   │ 110.0: reg(1,1)imm(4,4)│ 100.0: mem(7)    │
///  │            │   │ 110.1: reg(1,1)imm(8,8)│ 100.1: mem(11)   │
///  │            │   │ 111.x: opcode(1)       │ 101.0: mem(7,7)  │
///  │            │   │                        │ 101.1: mem(11,11)│
///  │            │   │                        │ 110.x: off(0)ExtC│
///  │            │   │                        │ 111.x: off(0)ExtV│
///  └────────────┴───┴────────────────────────┴──────────────────┘
/// 
///  flag:
///    0. bits mode: 0: 32-bits, 1: 64-bits
///    1. sign mode: 0: signed,  1: unsigned
///    You can see such as: (`_i32`, `_u32`, `_i64`, `_u64`) in insn name.
/// 
///  decode:
///    -> check opcode: if AAA is 111, append two bytes else append one byte
///    -> check flag: get A/B field length and parse as operands
///    -> if BBB/VVV is 110/111: read one more byte and check ExtC/ExtV flag, 
///         extend length and repeat (if AAA is 111, append to opcode)
///    -> if BBB/VVV is not 110/111, read end
///    -> match opcode and operands
///    
/// 
///  encode:
///    -> you should encode opcode and all flags.
///    -> then you can fill the operands to blank bytes.
/// 
///  ┌────────────────┬───────────┬──────────────────────────────────┐
///  │   ExtC flag    │  A field  │             B field              │
///  ├────────────────┼───────────┼──────────────────────────────────┤
///  │       1        │    ???    │               ???                │
///  ├────────────────┼───────────┼──────────────────────────────────┤
///  │  000 000 00    │    ...    │ 000.x: (110->000) off(0)         │
///  │  ─── ─── ──    │   Same    │ 001.z: imm(4)   / imm(8)         │
///  │  BBB AAA MM    │    as     │ 010.z: imm(4,4) / imm(8,8)       │
///  │                │  A field  │ ... (Same as B field)            │
///  │                │           │ 110.x: (110->110) off(0)ExtC     │
///  │                │           │ 111.x: (110->111) off(0)ExtV     │
///  └────────────────┴───────────┴──────────────────────────────────┘
/// 
///  Extension Constant(Same as A&B field):
///    -> find in first Byte 0b110 in forward flag.
///    -> Read 1 more Byte and check flag for length of fields.
///    -> MM: Mem accessing enhance mode, 00: 8-byte, 01: 16-byte, 10: 32-byte, 11: 64-byte.
/// 
///  ┌────────────────┬───────────────────────────────────────────┐
///  │   ExtV flag    │               ExtV Field                  │
///  ├────────────────┼───────────────────────────────────────────┤
///  │       1        │                   ???                     │
///  ├────────────────┼───────────────────────────────────────────┤
///  │   000  00000   │   000.x: (111->000) off(0)                │
///  │   ───  ─────   │   001.x: vec,len                          │
///  │   VVV  index   │   010.x: vec,vec,len                      │
///  │                │   ... (User define Operand Pattern)       │
///  │        00002   │   110.x: (111->110) off(0)ExtC            │
///  │  (ExtV `VEC`)  │   111.x: (111->111) off(0)ExtV            │
///  └────────────────┴───────────────────────────────────────────┘
/// 
///  Extension Variable(User define field):
///    -> find in first Byte 0b111 in forward flag.
///    -> Read 1 more Byte and check ExtV table index.
///    -> According to index deal with operands.
/// 
/// 
/// ```
/// 
/// - i386/x86_64 format:
/// ```txt
/// (Variable-length/Byte): MAX 15 Bytes. --> Little Endian View.
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
///  Default Env:        64bit    32bit   16bit
///    - address-size:    64       32      16
///    - operand-size:    32       32      16
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
///    0~2. r/m - As Direct/Indirect operand(E): Reg/Mem.
///    3~5. r/op - As Reg ref(G), or as 3-bit opcode extension.
///    6~7. mod - 0b00: [base], 0b01: [base + disp8], 0b10: [base + disp32], 0b11: Reg.
///    Such as: ADD ecx, esi (po=0x01), set ModR/M: 0b11 110(esi) 001(ecx)=0xf1.
///    Get (Insn=0x01 f1)
///  Prefixs(Legacy):
///    - instruction prefix
///    - address-size override prefix: 0x67(Default: 32 -> 16)
///    - operand-size override prefix: 0x66(Default: 32 -> 16)
///    - segment override prefix: 0x2e(CS) 0x3e(DS) 0x26(ES) 0x36(SS) 0x64(FS) 0x65(GS)
///    - repne/repnz prefix: 0xf2 0xf3
///    - lock prefix: 0xf0
///    Such as: MOV r/m32, r32 (po=0x89), set opr-prefix: 0x66
///    Get MOV r/m16, r16 (Insn=0x66 89 ..)
///  SIB:
///    0~2. base
///    3~5. index
///    6~7. scale - 0b00: [idx], 0b01: [idx*2], 0b10: [idx*4], 0b11: [idx*8].
///  Rex Prefix(Only x86_64):
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// ### Instruction flag
    /// ```txt
    /// (0-7):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ ├─┴─┘   (0-2) 000 is 8-bit , 001 is 16-bit, 010 is 32-bit , 011 is 64-bit
    /// │ │ │ │ │ └────── (0-2) 100 is 80-bit, 101 is 96-bit, 110 is 128-bit, 111 is 256-bit
    /// │ │ │ │ └──────── (3) 0 is little-endian, 1 is big-endian
    /// │ │ │ │         ┌ (4-7) 0000: COND_NO , 0001: COND_ALL, 0010: COND_EQ , 0011: COND_NE
    /// │ │ │ │         │ (4-7) 0100: COND_LT , 0101: COND_GE , 0110: COND_LE , 0111: COND_GT
    /// ├─┴─┴─┘         │ (4-7) 1000: COND_LTU, 1001: COND_GEU, 1010: COND_LEU, 1011: COND_GTU
    /// └───────────────┴ (4-7) 
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
    pub addr: Option<Value>,
    pub label: Option<String>,
    pub opc : Opcode,
    pub opr : Vec<Operand>,
    pub opb : &'static str,
    pub code : Value,
    pub arch : &'static Arch,
    pub regfile : Rc<RefCell<RegFile>>,
    pub is_applied : bool,
    /// encode func
    pub enc : Option<fn(&mut Instruction, Vec<Operand>) -> Instruction>,
}

impl Instruction {

    // Init Opcode POOL
    thread_local! {
        /// IR Insn pool (Shared Global)
        // pub static INSN_POOL: Rc<RefCell<Vec<Rc<RefCell<Instruction>>>>> = Rc::new(RefCell::new(Vec::new()));

        pub static INSN_POOL: Rc<RefCell<HashMap<(&'static Arch, &'static str), Rc<RefCell<Instruction>>>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    /// Get Undef Insn
    pub fn undef() -> Instruction {
        Instruction {
            flag: 0,
            addr: None,
            label: None,
            opc: Opcode::new(".insn", Vec::new(), "Undef"),
            opr: Vec::new(),
            opb: "",
            code: Value::bit(0, 0),
            arch: &EVO_ARCH,
            regfile: Rc::new(RefCell::new(RegFile::new(&EVO_ARCH))),
            is_applied: false,
            enc: None,
        }
    }

    /// Define Instruction Temp and add to pool
    pub fn def(arch: &'static Arch, name: &'static str, flag: u16, syms: Vec<u16>, ty: &'static str, opb: &'static str) -> Instruction {
        let opc = Opcode::new(name, syms, ty);
        let insn = Instruction {
            flag: flag,
            addr: None,
            label: None,
            opc: opc.clone(),
            opr: Vec::new(),
            opb,
            code: Value::from_string(opb),
            arch,
            regfile: Rc::new(RefCell::new(RegFile::new(arch))),
            is_applied: false,
            enc: Self::encode_pool_init(arch)
        };
        // add to pool
        Instruction::insn_pool_push(arch, insn);
        // return
        Instruction::insn_pool_nget(arch, name).borrow().clone()
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
    pub fn apply(arch: &'static Arch, name: &'static str, opr: Vec<Operand>) -> Instruction {
        let mut insn = Instruction::insn_pool_nget(arch, name).borrow().clone();
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

    pub fn bits_mode(&self) -> usize {
        match self.flag & 0b0111 {
            0b000 => 8,
            0b001 => 16,
            0b010 => 32,
            0b011 => 64,
            0b100 => 80,
            0b101 => 96,
            0b110 => 128,
            0b111 => 256,
            _ => 32
        }
    }

    pub fn sign_mode(&self) -> usize {
        match self.flag & 0b0001_0000_0000_0000 {
            INSN_SIG => 0,
            INSN_USD => 1,
            _ => 0
        }
    }

    pub fn is_32bit(&self) -> bool {
        (self.flag & 0b0111) == BIT32
    }

    pub fn is_64bit(&self) -> bool {
        (self.flag & 0b0111) == BIT64
    }

    pub fn is_signed(&self) -> bool {
        (self.flag & 0b0001_0000_0000_0000) == INSN_SIG
    }

    pub fn is_unsigned(&self) -> bool {
        (self.flag & 0b0001_0000_0000_0000) == INSN_USD
    }

    // ================= Instruction.insn_pool ================== //

    /// delete Insn from pool
    pub fn insn_pool_del(arch: &'static Arch, name: &'static str) {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().remove(&(arch, name)));
    }

    /// get Insn from pool
    pub fn insn_pool_nget(arch: &'static Arch, name: &str) -> Rc<RefCell<Instruction>> {
        Self::INSN_POOL.with(|pool| pool.borrow().get(&(arch, name)).unwrap().clone()) 
    }

    /// Set Insn from pool
    pub fn insn_pool_nset(arch: &'static Arch, name: &'static str, insn: Instruction) {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().insert((arch, name), Rc::new(RefCell::new(insn))));
    }

    /// Insert Insn from pool
    pub fn insn_pool_push(arch: &'static Arch, insn: Instruction) {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().insert((arch, insn.name()), Rc::new(RefCell::new(insn))));
    }

    /// Clear Temp Insn Pool
    pub fn insn_pool_clr() {
        Self::INSN_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Get info of Insn Pool
    pub fn insn_pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Instructions (Num = {}):\n", Self::insn_pool_size()));
        Self::INSN_POOL.with(|pool| 
            for (arch, name) in pool.borrow().keys() {
                let insn = Self::insn_pool_nget(arch, name).borrow().clone();
                info.push_str(&format!("- {:<52}   {} ({})\n", insn.info(), insn.ty(), insn.opb));
            }
        );
        info
    }

    /// Get size of Insn Pool
    pub fn insn_pool_size() -> usize {
        Self::INSN_POOL.with(|pool| pool.borrow().len())
    }

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

    pub fn hex(&self, index: usize, byte_num: i32, big_endian: bool) -> String {
        self.code.hex(index, byte_num, big_endian)
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

    /// set label
    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
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
            log_warning!("Code not applied: {} ", self.opc.name());
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
        if self.label.is_some() {
            info.push_str(&format!("{:<8}: ", self.label.as_ref().unwrap()));
        } else {
            info.push_str(&format!("{:<9} ", ""));
        }

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
    pub fn from_string(arch: &'static Arch, str: &str) -> Instruction {
        // 1. Deal with string
        let mut str = str.trim();
        let mut label = None;
        // Deal with `label: [insn]`: if contains `:` and before `:` is identifier
        if str.contains(':') {
            let mut part = str.splitn(2, ':');
            let label_str = part.next().unwrap().trim();
            let insn = part.next().unwrap().trim();
            if Value::is_ident(label_str) {
                str = insn;
                label = Some(label_str.to_string());
            }
        }

        // 2. Divide in first space and Get Opcode: `[opc] [opr1], [opr2], ...`
        let mut part = str.splitn(2, ' ');
        // 3. Find Insn from pool
        let res = Instruction::insn_pool_nget(arch, part.next().unwrap().trim());
        // 4. Collect extra part: Divide by `,` and Get Operands str
        let pn = part.next();
        let extra_part = if pn.is_none() { 
            ""
        } else {
            pn.unwrap()
        };
        // let extra_part = part.next().unwrap();
        let opr = extra_part.split(',').collect::<Vec<_>>();
        // 5. Get Operands by sym
        let opr_sym = res.borrow().syms();
        let mut opr_vec = Vec::new();
        for i in 0..opr_sym.len() {
            let r;
            if i < opr.len() {
                r = Operand::from_string(opr_sym[i], opr[i].trim());
            } else {
                r = Operand::off();
            }
            opr_vec.push(r);
        }
        // 5. Encode Instruction
        let mut res = res.borrow_mut().encode(opr_vec);
        res.set_label(label);
        res
    }
}


impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}




