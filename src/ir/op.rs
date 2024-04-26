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
use std::collections::HashMap;

use crate::log_error;
use crate::util::log::Span;
use crate::ir::val::IRValue;
use crate::ir::ty::IRTypeKind;





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
        }
    }

    /// Get name of the operand
    pub fn name(&self) -> &'static str {
        match self {
            IROperandKind::Imm(_) => "<Imm>",
            IROperandKind::Reg(name, _) => name,
            IROperandKind::Mem(_, _, _, _) => "<Mem>",
            IROperandKind::Label(name, _) => name,
        }
    }

    /// Get String of the operand like: `val : kind`
    pub fn to_string(&self) -> String {
        match self {
            IROperandKind::Imm(val) =>  format!("{}: {}", val.get_u32(0) , val.kind()),
            IROperandKind::Reg(name, val) => format!("{}: {}", name, val.kind()),
            IROperandKind::Mem(base, idx, scale, disp) => format!("[{} + {} * {} + {}]: {}", base.kind(), idx.kind(), scale.kind(), disp.kind() , self.val().kind()),
            IROperandKind::Label(name, val) => format!("{}: {}", name, val.kind()),
        }
    }

    /// Get value of the operand
    pub fn val(&self) -> IRValue {
        match self {
            IROperandKind::Imm(val) => val.clone(),
            IROperandKind::Reg(_, val) => val.clone(),
            IROperandKind::Mem(base, idx, scale, disp) => IRValue::u32(base.get_u32(0) + idx.get_u32(0) * scale.get_u32(0) + disp.get_u32(0)),
            IROperandKind::Label(_, val) => val.clone(),
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

    /// New default Reg
    pub fn reg_dft() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Reg("<Reg>", IRValue::u32(0)))))
    }

    /// New default Imm
    pub fn imm_dft() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Imm(IRValue::u32(0)))))
    }

    /// New default Mem
    pub fn mem_dft() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Mem(IRValue::u32(0), IRValue::u32(0), IRValue::u32(0), IRValue::u32(0)))))
    }

    /// New default Label
    pub fn label_dft() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Label("<Label>", IRValue::u32(0)))))
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

    /// To String like: `val : kind`
    pub fn to_string(&self) -> String {
        self.0.borrow().to_string()
    }

}

impl fmt::Display for IROperand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0.borrow().to_string())
    }
}



// ============================================================================== //
//                              op::IROpcodeKind
// ============================================================================== //

/// `IROpcodeKind`: Kind of IR opcodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROpcodeKind {

    /// I-Type Opcode
    I(&'static str, Vec<i32>),

    /// R-Type Opcode
    R(&'static str, Vec<i32>),

    /// J-Type Opcode
    J(&'static str, Vec<i32>),

}


impl IROpcodeKind {

    /// Get Name of OpcodeKind
    pub fn name(&self) -> &'static str {
        match self {
            IROpcodeKind::I(name, _) => name,
            IROpcodeKind::R(name, _) => name,
            IROpcodeKind::J(name, _) => name,
        }
    }

    /// Get Type name of OpcodeKind
    pub fn ty(&self) -> &'static str {
        match self {
            IROpcodeKind::I(_, _) => "I",
            IROpcodeKind::R(_, _) => "R",
            IROpcodeKind::J(_, _) => "J",
        }
    }

    /// Get Symbols of OpcodeKind
    pub fn syms(&self) -> Vec<i32> {
        match self {
            IROpcodeKind::I(_, syms) => syms.clone(),
            IROpcodeKind::R(_, syms) => syms.clone(),
            IROpcodeKind::J(_, syms) => syms.clone(),
        }
    }

    /// Set Symbols of OpcodeKind
    pub fn set_syms(&mut self, syms: Vec<i32>) {
        match self {
            IROpcodeKind::I(_, _) => {
                *self = IROpcodeKind::I(self.name(), syms);
            },
            IROpcodeKind::R(_, _) => {
                *self = IROpcodeKind::R(self.name(), syms);
            },
            IROpcodeKind::J(_, _) => {
                *self = IROpcodeKind::J(self.name(), syms);
            },
        }
    }

    /// Check Symbols
    pub fn check_syms (&self, syms: Vec<i32>) -> bool {
        match self {
            IROpcodeKind::I(_, _) => {
                self.syms() == syms
            },
            IROpcodeKind::R(_, _) => {
                self.syms() == syms
            },
            IROpcodeKind::J(_, _) => {
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
            IROpcodeKind::J(name, syms) => {
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
            "I" => IROpcode(RefCell::new(IROpcodeKind::I(name, syms))),
            "R" => IROpcode(RefCell::new(IROpcodeKind::R(name, syms))),
            "J" => IROpcode(RefCell::new(IROpcodeKind::J(name, syms))),
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


// ============================================================================== //
//                                op::Instruction
// ============================================================================== //

/// `IRInsn`: IR instruction
#[derive(Debug, Clone, PartialEq)]
pub struct IRInsn {
    opc : IROpcode,
    opr : Vec<IROperand>,
    opb : &'static str,
    byt : IRValue,
    is_applied : bool,
}

impl IRInsn {

    // Init Opcode POOL
    thread_local! {
        static INSN_POOL: RefCell<HashMap<&'static str, IRInsn>> = RefCell::new(HashMap::new());
    }

    /// Define IRInsn Temp
    pub fn def(name: &'static str, syms: Vec<i32>, ty: &'static str, opb: &'static str) -> IRInsn {
        let opc = IROpcode::new(name, syms, ty);
        let insn = IRInsn {
            opc: opc.clone(),
            opr: Vec::new(),
            opb,
            byt: IRValue::from_string(opb),
            is_applied: false,
        };
        // add to pool
        IRInsn::set(name, insn);
        // return
        IRInsn::get(name)
    }

    /// apply temp to IRInsn
    pub fn apply(name: &'static str, opr: Vec<IROperand>) -> IRInsn {
        let mut insn = IRInsn::get(name);
        insn.apply_opr(opr)
    }

    // ==================== IRInsn.pool ==================== //

    /// delete Insn from pool
    pub fn del(name: &'static str) {
        IRInsn::INSN_POOL.with(|pool| {
            pool.borrow_mut().remove(name);
        })
    }

    /// get Insn from pool
    pub fn get(name: &'static str) -> IRInsn {
        // Find from pool
        IRInsn::INSN_POOL.with(|pool| {
            pool.borrow().get(name).unwrap().clone()
        })
    }

    /// Set Insn from pool
    pub fn set(name: &'static str, insn: IRInsn) {
        IRInsn::INSN_POOL.with(|pool| {
            pool.borrow_mut().insert(name, insn);
        })
    }

    // ==================== IRInsn.get ===================== //
    
    /// Get byte size of IRInsn
    pub fn size(&self) -> usize {
        self.byt.size()
    }

    /// Show binary byte string of IRInsn
    pub fn bin(&self, index: usize, byte_num: i32, big_endian: bool) -> String {
        self.byt.bin(index, byte_num, big_endian)
    }

    /// Show type of IRInsn
    pub fn ty(&self) -> &'static str {
        self.opc.ty()
    }

    /// Get String of Opcode + Operand Syms
    pub fn info(&self) -> String {
        format!("{}", self.opc.to_string())
    }

    /// Get binary define str of insn
    pub fn opbit(&self) -> &'static str {
        self.opb
    }
    

    /// To string: `[opc] [opr1] : [sym1], [opr2] : [sym2], ...`
    pub fn to_string(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("{:<6} ", self.opc.name()));
        for opr in self.opr.iter() {
            info.push_str(&format!("{}", opr.to_string()));
            // if iter not last push `,`
            if opr != self.opr.last().unwrap() {
                info.push_str(&format!(", "));
            }
        }
        info
    }


    // ==================== IRInsn.sym ===================== //

    /// Check IROperand Syms
    pub fn check_syms(&self, opr: Vec<IROperand>) -> bool {
        self.opc.check_syms(opr.iter().map(|x| x.sym()).collect())
    }

    /// Reset IROperand Syms
    pub fn set_syms(&mut self, opr: Vec<IROperand>) {
        self.opc.set_syms(opr.iter().map(|x| x.sym()).collect());
    }


    // ==================== IRInsn.opr ===================== //

    /// apply IROperand to IROpcode
    pub fn apply_opr(&mut self, opr: Vec<IROperand>) -> IRInsn {
        // Check syms
        if self.check_syms(opr.clone()) {
            self.opr = opr;
            self.is_applied = true;
            self.clone()
        } else {
            // Error
            log_error!("Apply operands failed: {} ", self.opc.name());
            // Revert
            IRInsn::def(self.opc.name(), opr.iter().map(|x| x.sym()).collect(), self.opc.ty(), self.opb)
        }
    }
}


impl fmt::Display for IRInsn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

