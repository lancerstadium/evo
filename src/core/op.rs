//! `evo::ir::op`: Opcodes and operands definition in the IR
//! 
//! 


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::fmt::{self};
use std::cmp;
use std::rc::Rc;
use std::cell::RefCell;

use crate::arch::info::{BIT16, BIT32, BIT64, BIT8, LITTLE_ENDIAN};
use crate::{log_error, log_warning};
use crate::util::log::Span;
use crate::core::val::Value;
use crate::core::ty::TypesKind;
use crate::core::insn::RegFile;


// ============================================================================== //
//                                  Const
// ============================================================================== //

pub const OPR_UND: u16 = 0b0000_0000;
pub const OPR_IMM: u16 = 0b0000_0001;
pub const OPR_REG: u16 = 0b0000_0010;
pub const OPR_MEM: u16 = 0b0000_0100;
pub const OPR_LAB: u16 = 0b0000_1000;
pub const OPR_OFF: u16 = 0b0001_0000;

/// reg offset 0-byte
pub const REG_OFF0 : u16 = 0b0000_0000_0000;
/// reg offset 1-byte
pub const REG_OFF8 : u16 = 0b0001_0000_0000;
/// reg offset 2-byte
pub const REG_OFF16: u16 = 0b0010_0000_0000;
/// reg offset 3-byte
pub const REG_OFF24: u16 = 0b0011_0000_0000;
/// reg offset 4-byte
pub const REG_OFF32: u16 = 0b0100_0000_0000;
/// reg offset 5-byte
pub const REG_OFF40: u16 = 0b0101_0000_0000;
/// reg offset 6-byte
pub const REG_OFF48: u16 = 0b0110_0000_0000;
/// reg offset 7-byte
pub const REG_OFF56: u16 = 0b0111_0000_0000;
/// reg offset 8-byte
pub const REG_OFF64: u16 = 0b1000_0000_0000;
/// reg offset 9-byte
pub const REG_OFF72: u16 = 0b1001_0000_0000;
/// reg offset 10-byte
pub const REG_OFF80: u16 = 0b1010_0000_0000;
/// reg offset 11-byte
pub const REG_OFF88: u16 = 0b1011_0000_0000;
/// reg offset 12-byte
pub const REG_OFF96: u16 = 0b1100_0000_0000;
/// reg offset 13-byte
pub const REG_OFF104: u16 = 0b1101_0000_0000;
/// reg offset 14-byte
pub const REG_OFF112: u16 = 0b1110_0000_0000;
/// reg offset 15-byte
pub const REG_OFF120: u16 = 0b1111_0000_0000;
/// global reg: need to be bundled
pub const REG_GLOB: u16 = 0b0000_0000;
/// global reg: bundled
pub const REG_BUND: u16 = 0b0001_0000;
/// local reg: Using for local variable
pub const REG_LOCA: u16 = 0b0010_0000;
/// local reg: using for temp reg
pub const REG_TEMP: u16 = 0b0011_0000;




// ============================================================================== //
//                              op::OperandKind
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandKind {
    /// |  val  |
    Imm(Value),

    /// |  name*  |  idx  | flag |
    Reg(&'static str, Value, u16),

    /// Mem = [base + index * scale + disp]
    /// |  base  |  idx  |  scala  | disp  |
    Mem(Operand, Operand, Value, Value),

    /// |  nick*  |  pc  |
    Label(String, Value),

    /// Off this operand
    Off,

    /// Undefined
    Undef,
}



impl OperandKind {

    /// Get the kind of value
    pub fn kind(&self) -> &TypesKind{
        match self {
            OperandKind::Imm(val) => val.kind(),
            // Get Reg value's kind
            OperandKind::Reg(_, val, _) => val.kind(),
            OperandKind::Mem(_, _, _, disp) => disp.kind(),
            // Get Label ptr's kind
            OperandKind::Label(_, val) => val.kind(),
            OperandKind::Off => &TypesKind::None,
            OperandKind::Undef => &TypesKind::None,
        }
    }

    /// Get hex of the operand
    pub fn hex(&self) -> String {
        match self {
            OperandKind::Imm(val) => val.hex(0, -1, false),
            // Get Reg value's hex
            OperandKind::Reg(_, val, _) => val.hex(0, -1, false),
            OperandKind::Mem(_, _, _, _) => self.val().hex(0, -1, false),
            // Get Label ptr's hex
            OperandKind::Label(_, val) => val.hex(0, -1, false),
            OperandKind::Off => "".to_string(),
            OperandKind::Undef => "".to_string(),
        }
    }

    /// Get name of the operand: reg
    pub fn name(&self) -> &'static str {
        match self {
            OperandKind::Imm(_) => "<Imm>",
            OperandKind::Reg(name, _, _) => name,
            OperandKind::Mem(_, _, _, _) => "<Mem>",
            OperandKind::Label(_, _) => "<Lab>",
            OperandKind::Off => "<Off>",
            OperandKind::Undef => "<Und>",
        }
    }

    /// Get nick of the operand: label
    pub fn nick(&self) -> String {
        match self {
            OperandKind::Imm(_) => "<Imm>".to_string(),
            OperandKind::Reg(_, _, _) => "<Reg>".to_string(),
            OperandKind::Mem(_, _, _, _) => "<Mem>".to_string(),
            OperandKind::Label(nick, _) => nick.clone(),
            OperandKind::Off => "<Off>".to_string(),
            OperandKind::Undef => "<Und>".to_string(),
        }
    }

    /// Get String of the operand like: `val : kind`
    pub fn to_string(&self) -> String {
        match self {
            OperandKind::Imm(val) =>  format!("{}: {}", val.hex(0, -1, false) , val.kind()),
            OperandKind::Reg(name, val, _) => format!("{:>3}: {}", name, val.kind()),
            OperandKind::Mem(base, idx, scale, disp) => {
                let mut info = String::new();
                let mut is_gen = false;
                info.push('[');
                if RegFile::reg_poolr_is_in_all(&base.name()) {
                    info.push_str(base.name());
                    is_gen = true;
                }
                if RegFile::reg_poolr_is_in_all(&idx.name()) {
                    if is_gen { info.push('+'); }
                    info.push_str(idx.name());
                    is_gen = true;
                    if scale.get_byte(0) != 1 {
                        info.push('*');
                        info.push_str(scale.get_byte(0).to_string().as_str());
                    }
                }
                if disp.get_byte(0) != 0 {
                    if is_gen { info.push('+'); }
                    info.push_str(disp.hex(0, -1, false).as_str());
                }
                info.push(']');
                info.push_str(format!(": {}", self.val().kind()).as_str());
                info
            },
            OperandKind::Label(name, val) => format!("{}: {}", name, val.kind()),
            OperandKind::Off => "<Off>".to_string(),
            OperandKind::Undef => "<Und>".to_string(),
        }
    }

    /// Get value of the operand
    pub fn val(&self) -> Value {
        match self {
            OperandKind::Imm(val) => val.clone(),
            OperandKind::Reg(_, val, _) => val.clone(),
            OperandKind::Mem(base, idx, scale, disp) => Value::tuple(vec![base.val().clone(), idx.val().clone(), scale.clone(), disp.clone()]),
            OperandKind::Label(_, val) => val.clone(),
            OperandKind::Off => Value::bits(""),
            _ => Value::bits(""),
        }
    }

    /// Get symbol of the operandKind
    pub fn sym(&self) -> u16 {
        match self {
            OperandKind::Imm(_) => OPR_IMM,
            OperandKind::Reg(_, _, _) => OPR_REG,
            OperandKind::Mem(_, _, _, _) => OPR_MEM,
            OperandKind::Label(_, _) => OPR_LAB,
            OperandKind::Off => OPR_OFF,
            _ => OPR_UND,
        }
    }

    /// Get sym string
    pub fn sym_str(sym: u16) -> String {
        let mut info = String::new();
        info.push('<');
        if sym == OPR_UND {
            info.push_str("Und");
            info.push('>');
            return info;
        }
        let mut is_gen = false;
        if sym & OPR_IMM != 0 {
            info.push_str("Imm");
            is_gen = true;
        }
        if sym & OPR_REG != 0 {
            info.push_str(if is_gen {"/Reg"} else {"Reg"});
            is_gen = true;
        }
        if sym & OPR_MEM != 0 {
            info.push_str(if is_gen {"/Mem"} else {"Mem"});
            is_gen = true;
        }
        if sym & OPR_LAB != 0 {
            info.push_str(if is_gen {"/Lab"} else {"Lab"});
        }
        if sym & OPR_OFF != 0 {
            info.push_str(if is_gen {"/Off"} else {"Off"});
        }
        info.push('>');
        info
    }


}

impl fmt::Display for OperandKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ============================================================================== //
//                               op::Operand
// ============================================================================== //


/// `Operand`: Operands in the IR
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand(Rc<RefCell<OperandKind>>);

impl Operand {

    // ================== Operand.new ==================== //

    pub fn new(sym: u16) -> Self {
        match sym {
            OPR_IMM => Self::imm(Value::u32(0)),
            OPR_REG => Self::reg_def(),
            OPR_MEM => Self::mem(Self::reg_def(), Self::reg_def(), Value::u32(0), Value::u32(0)),
            OPR_LAB => Self::lab_def(),
            OPR_OFF => Self::off(),
            _ => Operand::imm(Value::u32(0)),
        }
    }

    /// New Operand Reg
    pub fn reg(name: &'static str, idx: Value, flag: u16) -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Reg(name, idx, flag))))
    }

    /// New Reg Default: `"<Reg>", -1: i8`
    pub fn reg_def() -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Reg("<Reg>", Value::i8(-1), BIT32 | LITTLE_ENDIAN))))
    }

    /// New Label Default: "<Lab>"
    pub fn lab_def() -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Label("<Lab>".to_string(), Value::i32(0)))))
    }

    /// New Operand Imm
    pub fn imm(val: Value) -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Imm(val))))
    }

    /// New Operand Mem
    pub fn mem(base: Operand, idx: Operand, scale: Value, disp: Value) -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Mem(base, idx, scale, disp))))
    }

    /// New Operand Label
    pub fn label(name: String, val: Value) -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Label(name, val))))
    }

    /// New Operand Off
    pub fn off() -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Off)))
    }

    /// New Operand Undef
    pub fn undef() -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Undef)))
    }

    /// info
    pub fn info(&self) -> String {
        match self.kind() {
            OperandKind::Imm(_) => self.to_string(),
            OperandKind::Reg(_, _, _) => {
                let idx_str = self.val().bin_scale(0, -1, true);
                format!("{:<9} [{:<2}:{:>2}]   {} ({})", self.to_string(), self.reg_scale() + self.reg_offset() * 8 - 1, self.reg_offset() * 8, self.reg_status(), idx_str)
            },
            OperandKind::Mem(_, _, _, _) => self.to_string(),
            OperandKind::Label(_, _) => self.to_string(),
            OperandKind::Off => self.to_string(),
            OperandKind::Undef => self.to_string(),
        }
    }

    // ================== Operand.get ==================== //

    /// Get Operand kind
    pub fn kind(&self) -> OperandKind {
        self.0.borrow_mut().clone()
    }

    /// Get Operand value
    pub fn val(&self) -> Value {
        self.0.borrow_mut().val()
    }

    /// Get Operand name
    pub fn name(&self) -> &'static str {
        self.0.borrow_mut().name()
    }

    /// Get symbol of the operand
    pub fn sym(&self) -> u16 {
        self.0.borrow_mut().sym()
    }

    /// Get symbol str
    pub fn sym_str(&self) -> String {
        OperandKind::sym_str(self.0.borrow_mut().sym())
    }

    /// Get reg
    pub fn get_reg(&self) -> usize {
        match self.kind() {
            OperandKind::Reg(_, val, _) => val.get_byte(0) as usize,
            _ => {
                log_warning!("Invalid Reg index: {}", self.to_string());
                return 0;
            }
        }
    }
    // ================== Operand.set ==================== //

    /// Set Operand value
    pub fn set_reg(&mut self, val: Value) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Reg(_, _, flag) => kind = OperandKind::Reg(self.name(), val, flag),
            _ => kind = OperandKind::Reg("<Reg>", val, BIT32 | LITTLE_ENDIAN),
        }
        self.0.replace(kind);
    }

    /// Set Imm value
    pub fn set_imm(&mut self, val: Value) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Imm(_) => kind = OperandKind::Imm(val),
            _ => kind = OperandKind::Imm(val),
        }
        self.0.replace(kind);
    }

    /// Set Mem value
    pub fn set_mem(&mut self, base: Operand, idx: Operand, scale: Value, disp: Value) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Mem(_, _, _, _) => kind = OperandKind::Mem(base, idx, scale, disp),
            _ => kind = OperandKind::Mem(base, idx, scale, disp),
        }
        self.0.replace(kind);
    }

    /// Set Label value
    pub fn set_label(&mut self, val: Value) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Label(nike, _) => kind = OperandKind::Label(nike, val),
            _ => kind = OperandKind::Label("<Lab>".to_string(), val),
        }
        self.0.replace(kind);
    }

    /// Get Mem value
    pub fn get_mem(&self) -> (usize, usize, u8, i32) {
        let kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Mem(base, idx, scale, disp) => {
                (base.val().get_byte(0) as usize, 
                idx.val().get_byte(0) as usize, 
                scale.get_byte(0) as u8, 
                disp.get_i32(0) as i32)
            },
            _ => {
                log_warning!("Invalid Mem value: {}", self.to_string());
                return (0, 0, 1, 0);
            },
        }
    }

    pub fn get_mem_base(&self) -> Operand {
        let kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Mem(base, _, _, _) => base.clone(),
            _ => Operand::undef(),
        }
    }

    pub fn is_8bit(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => (flag & 0b0011) == BIT8,
            _ => false,
        }
    }

    pub fn is_16bit(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => (flag & 0b0011) == BIT16,
            _ => false,
        }
    }

    pub fn is_32bit(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => (flag & 0b0011) == BIT32,
            _ => false,
        }
    }

    pub fn is_64bit(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => (flag & 0b0011) == BIT64,
            _ => false,
        }
    }

    pub fn reg_offset(&self) -> usize {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                ((flag & 0x0f00) >> 8) as usize
            }
            _ => 0,
        }
    }

    pub fn reg_scale(&self) -> usize {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                match flag & 0b0011 {
                    BIT8 => 8,
                    BIT16 => 16,
                    BIT32 => 32,
                    BIT64 => 64,
                    _ => 0,
                }
            }
            _ => 0,
        }
    }


    pub fn set_reg_bund(&self) {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                self.0.replace(OperandKind::Reg(self.name(), self.val(), (flag & 0b1111_1111_1100_1111) | REG_BUND));
            }
            _ => {},
        }
    }

    pub fn set_reg_global(&self) {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                self.0.replace(OperandKind::Reg(self.name(), self.val(), (flag & 0b1111_1111_1100_1111) | REG_GLOB));
            }
            _ => {},
        }
    }

    pub fn set_reg_local(&self) {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                self.0.replace(OperandKind::Reg(self.name(), self.val(), (flag & 0b1111_1111_1100_1111) | REG_LOCA));
            }
            _ => {},
        }
    }

    pub fn set_reg_temp(&self) {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                self.0.replace(OperandKind::Reg(self.name(), self.val(), (flag & 0b1111_1111_1100_1111) | REG_TEMP));
            }
            _ => {},
        }
    }

    pub fn reg_status(&self) -> String {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                match flag & 0b0011_0000 {
                    REG_GLOB => "G",
                    REG_BUND => "B",
                    REG_TEMP => "T",
                    REG_LOCA => "L",
                    _ => "",
                }.to_string()
            }
            _ => "".to_string(),
        }
    }

    pub fn is_reg_bund(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                (flag & 0b0011_0000) == REG_BUND
            },
            _ => false,
        }
    }

    pub fn is_reg_global(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                (flag & 0b0011_0000) == REG_GLOB
            },
            _ => false,
        }
    }

    pub fn is_reg_local(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                (flag & 0b0011_0000) == REG_LOCA
            },
            _ => false,
        }
    }

    pub fn is_reg_temp(&self) -> bool {
        match self.kind() {
            OperandKind::Reg(_, _, flag) => {
                (flag & 0b0011_0000) == REG_TEMP
            },
            _ => false,
        }
    }


    // ================== Operand.label ================== //

    pub fn label_nick(&self) -> String {
        match self.kind() {
            OperandKind::Label(nick, _) => nick.to_string(),
            _ => "".to_string(),
        }
    }

    pub fn label_pc(&self) -> Value {
        match self.kind() {
            OperandKind::Label(_, pc) => pc,
            _ => Value::i32(0),
        }
    }

    // ================== Operand.str ==================== //

    /// To String like: `val: kind`
    pub fn to_string(&self) -> String {
        self.0.borrow_mut().to_string()
    }

    /// From String like: `val: {kind}`
    /// ### Lookup Address
    /// Params:
    /// 1. base/idx: Reg field
    /// 2. scale: b'8 = (**b**/h/w/d: 1/2/4/8)
    /// 3. disp : i32
    /// #### 1 Direct Address
    /// ```txt
    /// [<Imm>]
    ///  disp
    /// ```
    /// #### 2 Indirect Address
    /// ```txt
    /// [<Reg>]
    ///  base
    /// ```
    /// #### 3 Base + Disp Address
    /// ```txt
    /// [<Reg> + <Imm>]
    ///   base    disp
    /// ```
    /// #### 4 Index Address
    /// ```txt
    /// [<Reg> + <Reg>]
    ///   base    idx
    /// 
    /// [<Reg> + <Reg> + <Imm>]
    ///   base    idx     disp
    /// ```
    /// #### 5 Scale index Address
    /// ```txt
    /// [<Reg> * <Imm>]
    ///   idx    scale
    /// 
    /// [<Reg> * <Imm> + <Imm>]
    ///   idx    scale    disp
    /// 
    /// [<Reg> + <Reg> * <Imm>]
    ///   base    idx    scale
    /// 
    /// [<Reg> + <Reg> * <Imm> + <Imm>]
    ///   base    idx    scale    disp
    /// ```
    pub fn from_string(sym: u16, opr: &'static str) -> Self {
        let opr = opr.trim();
        // 0. if sym == OPR_UND, return
        if sym == OPR_UND {
            log_warning!("Undefined operand: {}", opr);
            return Self(Rc::new(RefCell::new(OperandKind::Undef)));
        }
        // 0. Deal with off
        if sym & OPR_OFF != 0 && opr == "" {
            return Operand::off();
        }
        // 1. Deal with reg
        if (sym & OPR_REG != 0) && RegFile::reg_poolr_is_in_all(opr) {
            return RegFile::reg_poolr_nget_all(opr).borrow_mut().clone();
        }
        // 2. Deal with mem: `[base + idx * scale + disp]`
        if (sym & OPR_MEM != 0) && opr.starts_with('[') && opr.ends_with(']') {
            // 2.1 del `[` and `]`
            let opr = opr.trim_matches('[').trim_matches(']');
            // 2.2 iter and deal with char
            let mut base = Self::reg_def();
            let mut idx = Self::reg_def();
            let mut scale = Value::u8(1);
            let mut disp = Value::i32(0);
            let mut info = String::new();
            let mut has_scale = false;
            let mut deal_scale = false;
            let mut deal_idx = false;
            let mut deal_base = false;
            for c in opr.chars() {
                match c {
                    ' ' => continue,
                    '+' => {
                        // check info is in reg pool: True find base/idx Value
                        if RegFile::reg_poolr_is_in_all( &info) {
                            if !deal_base {
                                base = RegFile::reg_poolr_nget_all( &info).borrow_mut().clone();
                                info.clear();
                                deal_base = true;
                            } else if !deal_idx {
                                idx = RegFile::reg_poolr_nget_all(&info).borrow_mut().clone();    
                                info.clear();
                                deal_idx = true;
                            }
                        } else {    // False find scale Value
                            scale = Value::from_string(&info);
                            let val = scale.get_u8(0);
                            // check if scale is [1, 2, 4, 8]
                            if val != 1 && scale.get_u8(0) != 2 && scale.get_u8(0) != 4 && scale.get_u8(0) != 8 {
                                log_warning!("Scale: {} is not in [1, 2, 4, 8]", scale.get_u8(0));
                                scale.set_u8(0, 1);
                            } else {
                                scale.set_u8(0, val);
                            }
                            // flush info
                            info.clear();
                            deal_scale = true;
                        }
                    },
                    '*' => {
                        // check info is in reg pool: True find idx Value
                        if RegFile::reg_poolr_is_in_all( &info) {
                            idx = RegFile::reg_poolr_nget_all(&info).borrow_mut().clone();
                            info.clear();
                            deal_idx = true;
                            has_scale = true;
                        }
                    },
                    // digital and english letter
                    '0'..='9' | 'a'..='z' | 'A'..='Z' => info.push(c),
                    _ => {
                        break;
                    }
                }
            }
            if !info.is_empty() {
                // check info is in reg pool: True find base/idx Value
                if RegFile::reg_poolr_is_in_all( &info) {
                    if !deal_base {
                        base = RegFile::reg_poolr_nget_all(&info).borrow_mut().clone();
                        info.clear();
                    }else {
                        idx = RegFile::reg_poolr_nget_all(&info).borrow_mut().clone();    
                        info.clear();
                    }
                } else {
                    if deal_idx && !deal_scale && has_scale {
                        scale = Value::from_string(&info);
                        let val = scale.get_u8(0);
                        // check if scale is [1, 2, 4, 8]
                        if val != 1 && scale.get_u8(0) != 2 && scale.get_u8(0) != 4 && scale.get_u8(0) != 8 {
                            log_warning!("Scale: {} is not in [1, 2, 4, 8]", scale.get_u8(0));
                            scale.set_u8(0, 1);
                        } else {
                            scale.set_u8(0, val);
                        }
                        info.clear();
                    } else {
                        disp = Value::from_string(&info);
                        info.clear();
                    }
                }
            }
            return Operand::mem(base, idx, scale, disp);
        }
        // 3. Deal with imm: if all is digital and english letter
        if sym & OPR_IMM != 0 {
            return Operand::imm(Value::from_string(opr));
        }
        // 4. Deal with label
        if (sym & OPR_LAB != 0) && Value::is_ident(opr) {
            return Operand::label(opr.to_string(), Value::i32(0));
        }
        // 5. Deal with undef
        log_warning!("Undefined operand: {}", opr);
        Operand::undef()
    }

}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}



// ============================================================================== //
//                              op::OpcodeKind
// ============================================================================== //

/// `OpcodeKind`: Kind of IR opcodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpcodeKind {

    /// R-Type Opcode
    R(&'static str, Vec<u16>),
    /// I-Type Opcode
    I(&'static str, Vec<u16>),
    /// S-Type Opcode
    S(&'static str, Vec<u16>),
    /// B-Type Opcode
    B(&'static str, Vec<u16>),
    /// U-Type Opcode
    U(&'static str, Vec<u16>),
    /// J-Type Opcode
    J(&'static str, Vec<u16>),

    /// X-Type Opcode
    X(&'static str, Vec<u16>),

    /// E-Type Opcode
    E(&'static str, Vec<u16>),

    /// Undef-Type Opcode
    Undef(&'static str, Vec<u16>),
}


impl OpcodeKind {

    /// Get Name of OpcodeKind
    pub fn name(&self) -> &'static str {
        match self {
            OpcodeKind::R(name, _) => name,
            OpcodeKind::I(name, _) => name,
            OpcodeKind::S(name, _) => name,
            OpcodeKind::B(name, _) => name,
            OpcodeKind::U(name, _) => name,
            OpcodeKind::J(name, _) => name,
            OpcodeKind::X(name, _) => name,
            OpcodeKind::E(name, _) => name,
            OpcodeKind::Undef(name, _) => name,
        }
    }

    /// Get Type name of OpcodeKind
    pub fn ty(&self) -> &'static str {
        match self {
            OpcodeKind::R(_, _) => "R",
            OpcodeKind::I(_, _) => "I",
            OpcodeKind::S(_, _) => "S",
            OpcodeKind::B(_, _) => "B",
            OpcodeKind::U(_, _) => "U",
            OpcodeKind::J(_, _) => "J",
            OpcodeKind::X(_, _) => "X",
            OpcodeKind::E(_, _) => "E",
            OpcodeKind::Undef(_, _) => "Undef",
        }
    }

    /// Get Symbols of OpcodeKind
    pub fn syms(&self) -> Vec<u16> {
        match self {
            OpcodeKind::R(_, syms) => syms.clone(),
            OpcodeKind::I(_, syms) => syms.clone(),
            OpcodeKind::S(_, syms) => syms.clone(),
            OpcodeKind::B(_, syms) => syms.clone(),
            OpcodeKind::U(_, syms) => syms.clone(),
            OpcodeKind::J(_, syms) => syms.clone(),
            OpcodeKind::X(_, syms) => syms.clone(),
            OpcodeKind::E(_, syms) => syms.clone(),
            OpcodeKind::Undef(_, syms) => syms.clone(),
        }
    }

    /// Set Symbols of OpcodeKind
    pub fn set_syms(&mut self, syms: Vec<u16>) {
        match self {
            OpcodeKind::R(_, _) => {
                *self = OpcodeKind::R(self.name(), syms);
            },
            OpcodeKind::I(_, _) => {
                *self = OpcodeKind::I(self.name(), syms);
            },
            OpcodeKind::S(_, _) => {
                *self = OpcodeKind::S(self.name(), syms);
            }
            OpcodeKind::B(_, _) => {
                *self = OpcodeKind::B(self.name(), syms);
            }
            OpcodeKind::U(_, _) => {
                *self = OpcodeKind::U(self.name(), syms);
            }
            OpcodeKind::J(_, _) => {
                *self = OpcodeKind::J(self.name(), syms);
            },
            OpcodeKind::X(_, _) => {
                *self = OpcodeKind::X(self.name(), syms);
            },
            OpcodeKind::E(_, _) => {
                *self = OpcodeKind::E(self.name(), syms);
            }
            OpcodeKind::Undef(_, _) => {
                *self = OpcodeKind::J(self.name(), syms);
            },
        }
    }

    /// Check Symbols
    pub fn check_syms(&self, syms: Vec<u16>) -> bool {
        match self {
            OpcodeKind::R(_, _) |
            OpcodeKind::I(_, _) |
            OpcodeKind::S(_, _) |
            OpcodeKind::B(_, _) |
            OpcodeKind::U(_, _) |
            OpcodeKind::J(_, _) |
            OpcodeKind::X(_, _) |
            OpcodeKind::E(_, _) => {
                // Get every sym and check
                let mut is_same = true;
                assert!(syms.len() == self.syms().len());
                for i in 0..syms.len() {
                    if self.syms()[i] != (syms[i] | self.syms()[i]) {
                        is_same = false;
                        break;
                    }
                }
                is_same
            }
            OpcodeKind::Undef(_, _) => {
                self.syms() == syms
            },
        }
    }

    /// To string
    pub fn to_string(&self) -> String {
        match self {
            OpcodeKind::I(name, syms) |
            OpcodeKind::R(name, syms) |
            OpcodeKind::S(name, syms) |
            OpcodeKind::B(name, syms) |
            OpcodeKind::U(name, syms) |
            OpcodeKind::J(name, syms) |
            OpcodeKind::X(name, syms) |
            OpcodeKind::E(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| OperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<10} {}", name, sym_str)
            },
            OpcodeKind::Undef(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| OperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<10} {}", name, sym_str)
            },
        }
    }

}

impl fmt::Display for OpcodeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ============================================================================== //
//                               op::Opcode
// ============================================================================== //


/// `Opcode`: IR opcodes
#[derive(Debug, Clone, Eq)]
pub struct Opcode(RefCell<OpcodeKind>);

impl Opcode {

    /// New Type Opcode
    pub fn new(name: &'static str, syms: Vec<u16>, ty: &'static str) -> Opcode {
        match ty {
            "R" => Opcode(RefCell::new(OpcodeKind::R(name, syms))),
            "I" => Opcode(RefCell::new(OpcodeKind::I(name, syms))),
            "S" => Opcode(RefCell::new(OpcodeKind::S(name, syms))),
            "B" => Opcode(RefCell::new(OpcodeKind::B(name, syms))),
            "U" => Opcode(RefCell::new(OpcodeKind::U(name, syms))),
            "J" => Opcode(RefCell::new(OpcodeKind::J(name, syms))),
            "X" => Opcode(RefCell::new(OpcodeKind::X(name, syms))),
            "E" => Opcode(RefCell::new(OpcodeKind::E(name, syms))),
            "Undef" => Opcode(RefCell::new(OpcodeKind::Undef(name, syms))),
            _ => {
                log_error!("Unknown type: {}", ty);
                Opcode(RefCell::new(OpcodeKind::I(name, syms)))
            },
        }
    }
    
    /// Get Name of Opcode
    pub fn name(&self) -> &'static str {
        self.0.borrow_mut().name()
    }

    /// Get Type name of Opcode
    pub fn ty(&self) -> &'static str {
        self.0.borrow_mut().ty()
    }

    /// Get Symbols of Opcode
    pub fn syms(&self) -> Vec<u16> {
        self.0.borrow_mut().syms()
    }

    /// Check syms of Opcode
    pub fn check_syms(&self, syms: Vec<u16>) -> bool {
        self.0.borrow_mut().check_syms(syms)
    }

    /// Get a refence of OpcodeKind
    pub fn kind(&self) -> OpcodeKind {
        self.0.borrow_mut().clone()
    }

    // =================== Opcode.set ==================== //

    /// Set Symbols of Opcode
    pub fn set_syms(&mut self, syms: Vec<u16>) {
        self.0.borrow_mut().set_syms(syms);
    }

    /// To String
    pub fn to_string(&self) -> String {
        self.0.borrow_mut().to_string()
    }

    
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl cmp::PartialEq for Opcode {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name() && self.kind() == other.kind()
    }
}


