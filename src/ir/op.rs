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

use crate::log_error;
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
    Mem(Value, Value, Value, Value),

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
            OperandKind::Mem(val, _, _, _) => val.size(),
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
            OperandKind::Mem(val, _, _, _) => val.kind(),
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
            OperandKind::Mem(base, idx, scale, disp) => format!("[{} + {} * {} + {}]: {}", base.kind(), idx.kind(), scale.kind(), disp.kind() , self.val().kind()),
            OperandKind::Label(name, val) => format!("{}: {}", name, val.kind()),
            OperandKind::Undef => "<Und>".to_string(),
        }
    }

    /// Get value of the operand
    pub fn val(&self) -> Value {
        match self {
            OperandKind::Imm(val) => val.clone(),
            OperandKind::Reg(_, val) => val.clone(),
            OperandKind::Mem(base, idx, scale, disp) => Value::u32(base.get_u32(0) + idx.get_u32(0) * scale.get_u32(0) + disp.get_u32(0)),
            OperandKind::Label(_, val) => val.clone(),
            _ => Value::i32(0),
        }
    }

    /// Get symbol of the operandKind
    /// 0: Imm, 1: Reg, 2: Mem, 3: Label
    pub fn sym(&self) -> i32 {
        match self {
            OperandKind::Imm(_) => 0,
            OperandKind::Reg(_, _) => 1,
            OperandKind::Mem(_, _, _, _) => 2,
            OperandKind::Label(_, _) => 3,
            _ => -1,
        }
    }

    /// Get sym string
    pub fn sym_str(sym: i32) -> &'static str {
        match sym {
            0 => "<Imm>",   // Imm
            1 => "<Reg>",   // Reg
            2 => "<Mem>",   // Mem
            3 => "<Lab>",   // Label
            _ => "<Und>",   // Undefined
        }
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

    pub fn new(sym: i32) -> Self {
        match sym {
            0 => Self::imm(Value::u32(0)),
            1 => Self::reg("<Reg>", Value::u5(0)),
            2 => Self::mem(Value::u32(0), Value::u32(0), Value::u32(0), Value::u32(0)),
            3 => Self::label("<Lab>", Value::u32(0)),
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
    pub fn mem(base: Value, idx: Value, scale: Value, disp: Value) -> Self {
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
        self.0.borrow().val()
    }

    /// Get Operand name
    pub fn name(&self) -> &'static str {
        self.0.borrow().name()
    }

    /// Get symbol of the operand
    /// 0: Imm, 1: Reg, 2: Mem, 3: Label
    pub fn sym(&self) -> i32 {
        self.0.borrow().sym()
    }

    /// Get symbol str
    pub fn sym_str(&self) -> &'static str {
        OperandKind::sym_str(self.0.borrow().sym())
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
        self.0.borrow().to_string()
    }

    /// From String like: `val: {kind}`
    pub fn from_string(sym: i32, opr: &'static str) -> Self {
        match sym {
            0 => {
                // Parse as Imm: `val`
                let val = Value::from_string(opr);
                Operand::imm(val)
            },
            1 => {
                // Parse as Reg: `name`, find in reg pool
                let name = opr.trim();
                Instruction::reg_pool_nget(name).borrow_mut().clone()
            },
            2 => {
                // Parse as Mem: `[base, idx, scale, disp]`
                let mut opr = opr.trim()[1..opr.len()-1].split(',').collect::<Vec<_>>();
                let base = Value::from_string(opr.remove(0).trim());
                let idx = Value::from_string(opr.remove(0).trim());
                let scale = Value::from_string(opr.remove(0).trim());
                let disp = Value::from_string(opr.remove(0).trim());
                Operand::mem(base, idx, scale, disp)
            },
            3 => {
                // Parse as Label: `Label: val`
                let mut opr = opr.split(':');
                let name = opr.next().unwrap().trim();
                let val = Value::from_string(opr.next().unwrap().trim());
                Operand::label(name, val)
            },
            _ => Operand::undef(),
        }
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
    R(&'static str, Vec<i32>),
    /// I-Type Opcode
    I(&'static str, Vec<i32>),
    /// S-Type Opcode
    S(&'static str, Vec<i32>),
    /// B-Type Opcode
    B(&'static str, Vec<i32>),
    /// U-Type Opcode
    U(&'static str, Vec<i32>),
    /// J-Type Opcode
    J(&'static str, Vec<i32>),

    /// Undef-Type Opcode
    Undef(&'static str, Vec<i32>),
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
            OpcodeKind::Undef(_, _) => "Undef",
        }
    }

    /// Get Symbols of OpcodeKind
    pub fn syms(&self) -> Vec<i32> {
        match self {
            OpcodeKind::R(_, syms) => syms.clone(),
            OpcodeKind::I(_, syms) => syms.clone(),
            OpcodeKind::S(_, syms) => syms.clone(),
            OpcodeKind::B(_, syms) => syms.clone(),
            OpcodeKind::U(_, syms) => syms.clone(),
            OpcodeKind::J(_, syms) => syms.clone(),
            OpcodeKind::Undef(_, syms) => syms.clone(),
        }
    }

    /// Set Symbols of OpcodeKind
    pub fn set_syms(&mut self, syms: Vec<i32>) {
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
            OpcodeKind::Undef(_, _) => {
                *self = OpcodeKind::J(self.name(), syms);
            },
        }
    }

    /// Check Symbols
    pub fn check_syms (&self, syms: Vec<i32>) -> bool {
        match self {
            OpcodeKind::R(_, _) => {
                self.syms() == syms
            },
            OpcodeKind::I(_, _) => {
                self.syms() == syms
            },
            OpcodeKind::S(_, _) => {
                self.syms() == syms
            }
            OpcodeKind::B(_, _) => {
                self.syms() == syms
            }
            OpcodeKind::U(_, _) => {
                self.syms() == syms
            }
            OpcodeKind::J(_, _) => {
                self.syms() == syms
            },
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
            OpcodeKind::J(name, syms) => {
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
    pub fn new(name: &'static str, syms: Vec<i32>, ty: &'static str) -> Opcode {
        match ty {
            "R" => Opcode(RefCell::new(OpcodeKind::R(name, syms))),
            "I" => Opcode(RefCell::new(OpcodeKind::I(name, syms))),
            "S" => Opcode(RefCell::new(OpcodeKind::S(name, syms))),
            "B" => Opcode(RefCell::new(OpcodeKind::B(name, syms))),
            "U" => Opcode(RefCell::new(OpcodeKind::U(name, syms))),
            "J" => Opcode(RefCell::new(OpcodeKind::J(name, syms))),
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
    pub fn syms(&self) -> Vec<i32> {
        self.0.borrow_mut().syms()
    }

    /// Check syms of Opcode
    pub fn check_syms(&self, syms: Vec<i32>) -> bool {
        self.0.borrow_mut().check_syms(syms)
    }

    /// Get a refence of OpcodeKind
    pub fn kind(&self) -> OpcodeKind {
        self.0.borrow_mut().clone()
    }

    // =================== Opcode.set ==================== //

    /// Set Symbols of Opcode
    pub fn set_syms(&mut self, syms: Vec<i32>) {
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


