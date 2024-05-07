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
use crate::ir::val::IRValue;
use crate::ir::ty::IRTypeKind;
use crate::ir::insn::IRInsn;





// ============================================================================== //
//                              op::IROperandKind
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROperandKind {
    /// |  val  |
    Imm(IRValue),

    /// |  name*  |  idx   |
    Reg(&'static str, IRValue),

    /// Mem = [base + index * scale + disp]
    /// |  base  |  idx  |  scala  | disp  |
    Mem(IRValue, IRValue, IRValue, IRValue),

    /// |  name*  |  addr  |
    Label(&'static str, IRValue),

    /// Undefined
    Undef,
}

impl IROperandKind {

    /// Get the size of the operand
    pub fn size(&self) -> usize {
        match self {
            IROperandKind::Imm(val) => val.size(),
            // Get Reg value's size
            IROperandKind::Reg(_, val) => val.size(),
            IROperandKind::Mem(val, _, _, _) => val.size(),
            // Get Label ptr's size
            IROperandKind::Label(_, val) => val.size(),
            IROperandKind::Undef => 0,
        }
    }

    /// Get the kind of value
    pub fn kind(&self) -> &IRTypeKind{
        match self {
            IROperandKind::Imm(val) => val.kind(),
            // Get Reg value's kind
            IROperandKind::Reg(_, val) => val.kind(),
            IROperandKind::Mem(val, _, _, _) => val.kind(),
            // Get Label ptr's kind
            IROperandKind::Label(_, val) => val.kind(),
            IROperandKind::Undef => &IRTypeKind::I32,
        }
    }

    /// Get hex of the operand
    pub fn hex(&self) -> String {
        match self {
            IROperandKind::Imm(val) => val.hex(0, -1, false),
            // Get Reg value's hex
            IROperandKind::Reg(_, val) => val.hex(0, -1, false),
            IROperandKind::Mem(_, _, _, _) => self.val().hex(0, -1, false),
            // Get Label ptr's hex
            IROperandKind::Label(_, val) => val.hex(0, -1, false),
            IROperandKind::Undef => "0".to_string(),
        }
    }

    /// Get name of the operand
    pub fn name(&self) -> &'static str {
        match self {
            IROperandKind::Imm(_) => "<Imm>",
            IROperandKind::Reg(name, _) => name,
            IROperandKind::Mem(_, _, _, _) => "<Mem>",
            IROperandKind::Label(name, _) => name,
            IROperandKind::Undef => "<Und>",
        }
    }

    /// Get String of the operand like: `val : kind`
    pub fn to_string(&self) -> String {
        match self {
            IROperandKind::Imm(val) =>  format!("{}: {}", val.bin_scale(0, -1, true) , val.kind()),
            IROperandKind::Reg(name, val) => format!("{:>3}: {}", name, val.kind()),
            IROperandKind::Mem(base, idx, scale, disp) => format!("[{} + {} * {} + {}]: {}", base.kind(), idx.kind(), scale.kind(), disp.kind() , self.val().kind()),
            IROperandKind::Label(name, val) => format!("{}: {}", name, val.kind()),
            IROperandKind::Undef => "<Und>".to_string(),
        }
    }

    /// Get value of the operand
    pub fn val(&self) -> IRValue {
        match self {
            IROperandKind::Imm(val) => val.clone(),
            IROperandKind::Reg(_, val) => val.clone(),
            IROperandKind::Mem(base, idx, scale, disp) => IRValue::u32(base.get_u32(0) + idx.get_u32(0) * scale.get_u32(0) + disp.get_u32(0)),
            IROperandKind::Label(_, val) => val.clone(),
            _ => IRValue::i32(0),
        }
    }

    /// Get symbol of the operandKind
    /// 0: Imm, 1: Reg, 2: Mem, 3: Label
    pub fn sym(&self) -> i32 {
        match self {
            IROperandKind::Imm(_) => 0,
            IROperandKind::Reg(_, _) => 1,
            IROperandKind::Mem(_, _, _, _) => 2,
            IROperandKind::Label(_, _) => 3,
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


impl fmt::Display for IROperandKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ============================================================================== //
//                               op::IROperand
// ============================================================================== //


/// `IROperand`: Operands in the IR
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IROperand(Rc<RefCell<IROperandKind>>);

impl IROperand {

    // ================== IROperand.new ==================== //

    pub fn new(sym: i32) -> Self {
        match sym {
            0 => Self::imm(IRValue::u32(0)),
            1 => Self::reg("<Reg>", IRValue::u5(0)),
            2 => Self::mem(IRValue::u32(0), IRValue::u32(0), IRValue::u32(0), IRValue::u32(0)),
            3 => Self::label("<Lab>", IRValue::u32(0)),
            _ => IROperand::imm(IRValue::u32(0)),
        }
    }

    /// New IROperand Reg
    pub fn reg(name: &'static str, val: IRValue) -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Reg(name, val))))
    }

    /// New IROperand Imm
    pub fn imm(val: IRValue) -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Imm(val))))
    }

    /// New IROperand Mem
    pub fn mem(base: IRValue, idx: IRValue, scale: IRValue, disp: IRValue) -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Mem(base, idx, scale, disp))))
    }

    /// New IROperand Label
    pub fn label(name: &'static str, val: IRValue) -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Label(name, val))))
    }

    /// New IROperand Undef
    pub fn undef() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Undef)))
    }

    /// info
    pub fn info(&self) -> String {
        match self.kind() {
            IROperandKind::Imm(_) => self.to_string(),
            IROperandKind::Reg(_, _) => {
                let idx_str = self.val().bin_scale(0, -1, true);
                format!("{:<9} ({})", self.to_string(), idx_str)
            },
            IROperandKind::Mem(_, _, _, _) => self.to_string(),
            IROperandKind::Label(_, _) => self.to_string(),
            IROperandKind::Undef => "<Und>".to_string(),
        }
    }

    // ================== IROperand.get ==================== //

    /// Get Operand kind
    pub fn kind(&self) -> IROperandKind {
        self.0.borrow_mut().clone()
    }

    /// Get Operand value
    pub fn val(&self) -> IRValue {
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
        IROperandKind::sym_str(self.0.borrow().sym())
    }

    // ================== IROperand.set ==================== //

    /// Set Operand value
    pub fn set_reg(&mut self, val: IRValue) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            IROperandKind::Reg(_, _) => kind = IROperandKind::Reg(self.name(), val),
            _ => kind = IROperandKind::Reg("<Und>", val),
        }
        self.0.replace(kind);
    }

    /// Set Imm value
    pub fn set_imm(&mut self, val: IRValue) {
        let mut kind = self.kind(); // Create a mutable copy of the value
        match kind {
            IROperandKind::Imm(_) => kind = IROperandKind::Imm(val),
            _ => kind = IROperandKind::Imm(IRValue::u32(0)),
        }
        self.0.replace(kind);
    }


    // ================== IROperand.str ==================== //

    /// To String like: `val: kind`
    pub fn to_string(&self) -> String {
        self.0.borrow().to_string()
    }

    /// From String like: `val: {kind}`
    pub fn from_string(sym: i32, opr: &'static str) -> Self {
        match sym {
            0 => {
                // Parse as Imm: `val`
                let val = IRValue::from_string(opr);
                IROperand::imm(val)
            },
            1 => {
                // Parse as Reg: `name`, find in reg pool
                let name = opr.trim();
                IRInsn::reg_pool_nget(name).borrow_mut().clone()
            },
            2 => {
                // Parse as Mem: `[base, idx, scale, disp]`
                let mut opr = opr.trim()[1..opr.len()-1].split(',').collect::<Vec<_>>();
                let base = IRValue::from_string(opr.remove(0).trim());
                let idx = IRValue::from_string(opr.remove(0).trim());
                let scale = IRValue::from_string(opr.remove(0).trim());
                let disp = IRValue::from_string(opr.remove(0).trim());
                IROperand::mem(base, idx, scale, disp)
            },
            3 => {
                // Parse as Label: `Label: val`
                let mut opr = opr.split(':');
                let name = opr.next().unwrap().trim();
                let val = IRValue::from_string(opr.next().unwrap().trim());
                IROperand::label(name, val)
            },
            _ => IROperand::undef(),
        }
    }

}

impl fmt::Display for IROperand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}



// ============================================================================== //
//                              op::IROpcodeKind
// ============================================================================== //

/// `IROpcodeKind`: Kind of IR opcodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROpcodeKind {

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


impl IROpcodeKind {

    /// Get Name of OpcodeKind
    pub fn name(&self) -> &'static str {
        match self {
            IROpcodeKind::R(name, _) => name,
            IROpcodeKind::I(name, _) => name,
            IROpcodeKind::S(name, _) => name,
            IROpcodeKind::B(name, _) => name,
            IROpcodeKind::U(name, _) => name,
            IROpcodeKind::J(name, _) => name,
            IROpcodeKind::Undef(name, _) => name,
        }
    }

    /// Get Type name of OpcodeKind
    pub fn ty(&self) -> &'static str {
        match self {
            IROpcodeKind::R(_, _) => "R",
            IROpcodeKind::I(_, _) => "I",
            IROpcodeKind::S(_, _) => "S",
            IROpcodeKind::B(_, _) => "B",
            IROpcodeKind::U(_, _) => "U",
            IROpcodeKind::J(_, _) => "J",
            IROpcodeKind::Undef(_, _) => "Undef",
        }
    }

    /// Get Symbols of OpcodeKind
    pub fn syms(&self) -> Vec<i32> {
        match self {
            IROpcodeKind::R(_, syms) => syms.clone(),
            IROpcodeKind::I(_, syms) => syms.clone(),
            IROpcodeKind::S(_, syms) => syms.clone(),
            IROpcodeKind::B(_, syms) => syms.clone(),
            IROpcodeKind::U(_, syms) => syms.clone(),
            IROpcodeKind::J(_, syms) => syms.clone(),
            IROpcodeKind::Undef(_, syms) => syms.clone(),
        }
    }

    /// Set Symbols of OpcodeKind
    pub fn set_syms(&mut self, syms: Vec<i32>) {
        match self {
            IROpcodeKind::R(_, _) => {
                *self = IROpcodeKind::R(self.name(), syms);
            },
            IROpcodeKind::I(_, _) => {
                *self = IROpcodeKind::I(self.name(), syms);
            },
            IROpcodeKind::S(_, _) => {
                *self = IROpcodeKind::S(self.name(), syms);
            }
            IROpcodeKind::B(_, _) => {
                *self = IROpcodeKind::B(self.name(), syms);
            }
            IROpcodeKind::U(_, _) => {
                *self = IROpcodeKind::U(self.name(), syms);
            }
            IROpcodeKind::J(_, _) => {
                *self = IROpcodeKind::J(self.name(), syms);
            },
            IROpcodeKind::Undef(_, _) => {
                *self = IROpcodeKind::J(self.name(), syms);
            },
        }
    }

    /// Check Symbols
    pub fn check_syms (&self, syms: Vec<i32>) -> bool {
        match self {
            IROpcodeKind::R(_, _) => {
                self.syms() == syms
            },
            IROpcodeKind::I(_, _) => {
                self.syms() == syms
            },
            IROpcodeKind::S(_, _) => {
                self.syms() == syms
            }
            IROpcodeKind::B(_, _) => {
                self.syms() == syms
            }
            IROpcodeKind::U(_, _) => {
                self.syms() == syms
            }
            IROpcodeKind::J(_, _) => {
                self.syms() == syms
            },
            IROpcodeKind::Undef(_, _) => {
                self.syms() == syms
            },
        }
    }

    /// To string
    pub fn to_string(&self) -> String {
        match self {
            IROpcodeKind::I(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
            IROpcodeKind::R(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
            IROpcodeKind::S(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
            IROpcodeKind::B(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
            IROpcodeKind::U(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
            IROpcodeKind::J(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
            IROpcodeKind::Undef(name, syms) => {
                // Get syms and to string
                let sym_str = syms.iter().map(|x| IROperandKind::sym_str(x.clone())).collect::<Vec<_>>().join(", ");
                format!("{:<6} {}", name, sym_str)
            },
        }
    }

}

impl fmt::Display for IROpcodeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ============================================================================== //
//                               op::IROpcode
// ============================================================================== //


/// `IROpcode`: IR opcodes
#[derive(Debug, Clone)]
pub struct IROpcode(RefCell<IROpcodeKind>);

impl IROpcode {

    /// New Type Opcode
    pub fn new(name: &'static str, syms: Vec<i32>, ty: &'static str) -> IROpcode {
        match ty {
            "R" => IROpcode(RefCell::new(IROpcodeKind::R(name, syms))),
            "I" => IROpcode(RefCell::new(IROpcodeKind::I(name, syms))),
            "S" => IROpcode(RefCell::new(IROpcodeKind::S(name, syms))),
            "B" => IROpcode(RefCell::new(IROpcodeKind::B(name, syms))),
            "U" => IROpcode(RefCell::new(IROpcodeKind::U(name, syms))),
            "J" => IROpcode(RefCell::new(IROpcodeKind::J(name, syms))),
            "Undef" => IROpcode(RefCell::new(IROpcodeKind::Undef(name, syms))),
            _ => {
                log_error!("Unknown type: {}", ty);
                IROpcode(RefCell::new(IROpcodeKind::I(name, syms)))
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
    pub fn kind(&self) -> IROpcodeKind {
        self.0.borrow_mut().clone()
    }

    // =================== IROpcode.set ==================== //

    /// Set Symbols of Opcode
    pub fn set_syms(&mut self, syms: Vec<i32>) {
        self.0.borrow_mut().set_syms(syms);
    }

    /// To String
    pub fn to_string(&self) -> String {
        self.0.borrow_mut().to_string()
    }

    
}

impl fmt::Display for IROpcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl cmp::PartialEq for IROpcode {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name() && self.kind() == other.kind()
    }
}


