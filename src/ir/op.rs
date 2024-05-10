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

use crate::{log_error, log_warning};
use crate::util::log::Span;
use crate::ir::val::Value;
use crate::ir::ty::TypesKind;
use crate::ir::insn::Instruction;





// ============================================================================== //
//                              op::OperandKind
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandKind {
    /// |  val  |
    Imm(Value),

    /// |  name*  |  idx   |
    Reg(&'static str, Value),

    /// Mem = [base + index * scale + disp]
    /// |  base  |  idx  |  scala  | disp  |
    Mem(Operand, Operand, Value, Value),

    /// |  name*  |  addr  |
    Label(&'static str, Value),

    /// Undefined
    Undef,
}




impl OperandKind {

    /// Get the size of the operand
    pub fn size(&self) -> usize {
        match self {
            OperandKind::Imm(val) => val.size(),
            // Get Reg value's size
            OperandKind::Reg(_, val) => val.size(),
            OperandKind::Mem(_, _, scale, _) => scale.get_byte(0) as usize,
            // Get Label ptr's size
            OperandKind::Label(_, val) => val.size(),
            OperandKind::Undef => 0,
        }
    }

    /// Get the kind of value
    pub fn kind(&self) -> &TypesKind{
        match self {
            OperandKind::Imm(val) => val.kind(),
            // Get Reg value's kind
            OperandKind::Reg(_, val) => val.kind(),
            OperandKind::Mem(_, _, _, disp) => disp.kind(),
            // Get Label ptr's kind
            OperandKind::Label(_, val) => val.kind(),
            OperandKind::Undef => &TypesKind::I32,
        }
    }

    /// Get hex of the operand
    pub fn hex(&self) -> String {
        match self {
            OperandKind::Imm(val) => val.hex(0, -1, false),
            // Get Reg value's hex
            OperandKind::Reg(_, val) => val.hex(0, -1, false),
            OperandKind::Mem(_, _, _, _) => self.val().hex(0, -1, false),
            // Get Label ptr's hex
            OperandKind::Label(_, val) => val.hex(0, -1, false),
            OperandKind::Undef => "0".to_string(),
        }
    }

    /// Get name of the operand
    pub fn name(&self) -> &'static str {
        match self {
            OperandKind::Imm(_) => "<Imm>",
            OperandKind::Reg(name, _) => name,
            OperandKind::Mem(_, _, _, _) => "<Mem>",
            OperandKind::Label(name, _) => name,
            OperandKind::Undef => "<Und>",
        }
    }

    /// Get String of the operand like: `val : kind`
    pub fn to_string(&self) -> String {
        match self {
            OperandKind::Imm(val) =>  format!("{}: {}", val.bin_scale(0, -1, true) , val.kind()),
            OperandKind::Reg(name, val) => format!("{:>3}: {}", name, val.kind()),
            OperandKind::Mem(base, idx, scale, disp) => format!("[{}+{}*{}+{}]: {}", base.name(), idx.name(), scale.get_byte(0), disp.hex(0, -1, true) , self.val().kind()),
            OperandKind::Label(name, val) => format!("{}: {}", name, val.kind()),
            OperandKind::Undef => "<Und>".to_string(),
        }
    }

    /// Get value of the operand
    pub fn val(&self) -> Value {
        match self {
            OperandKind::Imm(val) => val.clone(),
            OperandKind::Reg(_, val) => val.clone(),
            OperandKind::Mem(base, idx, scale, disp) => Value::tuple(vec![base.val().clone(), idx.val().clone(), scale.clone(), disp.clone()]),
            OperandKind::Label(_, val) => val.clone(),
            _ => Value::i32(0),
        }
    }

    /// Get symbol of the operandKind
    pub fn sym(&self) -> u16 {
        match self {
            OperandKind::Imm(_) => OPR_IMM,
            OperandKind::Reg(_, _) => OPR_REG,
            OperandKind::Mem(_, _, _, _) => OPR_MEM,
            OperandKind::Label(_, _) => OPR_LAB,
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
        info.push('>');
        info
    }


}

pub const OPR_UND: u16 = 0b0000_0000;
pub const OPR_IMM: u16 = 0b0000_0001;
pub const OPR_REG: u16 = 0b0000_0010;
pub const OPR_MEM: u16 = 0b0000_0100;
pub const OPR_LAB: u16 = 0b0000_1000;


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
            OPR_REG => Self::reg("<Reg>", Value::i8(-1)),
            OPR_MEM => Self::mem(Self::reg("<Reg>", Value::i8(-1)), Self::reg("<Reg>", Value::i8(-1)), Value::u32(0), Value::u32(0)),
            OPR_LAB => Self::label("<Lab>", Value::u32(0)),
            _ => Operand::imm(Value::u32(0)),
        }
    }

    /// New Operand Reg
    pub fn reg(name: &'static str, val: Value) -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Reg(name, val))))
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
    pub fn label(name: &'static str, val: Value) -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Label(name, val))))
    }

    /// New Operand Undef
    pub fn undef() -> Self {
        Self(Rc::new(RefCell::new(OperandKind::Undef)))
    }

    /// info
    pub fn info(&self) -> String {
        match self.kind() {
            OperandKind::Imm(_) => self.to_string(),
            OperandKind::Reg(_, _) => {
                let idx_str = self.val().bin_scale(0, -1, true);
                format!("{:<9} ({})", self.to_string(), idx_str)
            },
            OperandKind::Mem(_, _, _, _) => self.to_string(),
            OperandKind::Label(_, _) => self.to_string(),
            OperandKind::Undef => "<Und>".to_string(),
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

    // ================== Operand.set ==================== //

    /// Set Operand value
    pub fn set_reg(&mut self, val: Value) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Reg(_, _) => kind = OperandKind::Reg(self.name(), val),
            _ => kind = OperandKind::Reg("<Und>", val),
        }
        self.0.replace(kind);
    }

    /// Set Imm value
    pub fn set_imm(&mut self, val: Value) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            OperandKind::Imm(_) => kind = OperandKind::Imm(val),
            _ => kind = OperandKind::Imm(Value::u32(0)),
        }
        self.0.replace(kind);
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
            return Self(Rc::new(RefCell::new(OperandKind::Undef)));
        }
        // 1. Deal with reg
        if (sym & OPR_REG != 0) && Instruction::reg_pool_is_in(opr) {
            return Instruction::reg_pool_nget(opr).borrow_mut().clone();
        }
        // 2. Deal with mem: `[base + idx * scale + disp]`
        if (sym & OPR_MEM != 0) && opr.starts_with('[') && opr.ends_with(']') {
            // 2.1 del `[` and `]`
            let opr = opr.trim_matches('[').trim_matches(']');
            // 2.2 iter and deal with char
            let mut base = Self::reg("<Reg>", Value::i8(-1));
            let mut idx = Self::reg("<Reg>", Value::i8(-1));
            let mut scale = Value::u8(1);
            let mut disp = Value::i32(0);
            let mut info = String::new();
            let mut deal_scale = false;
            let mut deal_idx = false;
            let mut deal_base = false;
            for c in opr.chars() {
                match c {
                    ' ' => continue,
                    '+' => {
                        // check info is in reg pool: True find base/idx Value
                        if Instruction::reg_pool_is_in(&info) {
                            if !deal_base {
                                base = Instruction::reg_pool_nget(&info).borrow_mut().clone();
                                info.clear();
                                deal_base = true;
                            } else if !deal_idx {
                                idx = Instruction::reg_pool_nget(&info).borrow_mut().clone();    
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
                        if Instruction::reg_pool_is_in(&info) {
                            idx = Instruction::reg_pool_nget(&info).borrow_mut().clone();
                            info.clear();
                            deal_idx = true;
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
                if Instruction::reg_pool_is_in(&info) {
                    if !deal_base {
                        base = Instruction::reg_pool_nget(&info).borrow_mut().clone();
                        info.clear();
                    }else {
                        idx = Instruction::reg_pool_nget(&info).borrow_mut().clone();    
                        info.clear();
                    }
                } else {
                    if deal_idx && !deal_scale {
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
        if (sym & OPR_IMM != 0) && opr.chars().all(|c| c.is_ascii_digit() || c.is_ascii_alphabetic()) {
            return Operand::imm(Value::from_string(opr));
        }
        // 4. Deal with label
        if (sym & OPR_LAB != 0) && opr.chars().all(|c| c.is_ascii_digit() || c.is_ascii_alphabetic()) {
            return Operand::label(opr, Value::i32(0));
        }
        // 5. Deal with undef
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
            OpcodeKind::X(_, _) => {
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
            OpcodeKind::X(name, syms) => {
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
#[derive(Debug, Clone)]
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


