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
            IROperandKind::Imm(val) =>  format!("{}: {}", val.bin_scale(0, -1, true) , val.kind()),
            IROperandKind::Reg(name, val) => format!("{:>3}: {}", name, val.kind()),
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


    /// info
    pub fn info(&self) -> String {
        match self.kind() {
            IROperandKind::Imm(_) => self.to_string(),
            IROperandKind::Reg(_, val) => {
                let idx_str = self.val().bin_scale(0, -1, true);
                format!("{:<9} ({:>2}: {})", self.to_string(), val.to_string(), idx_str)
            },
            IROperandKind::Mem(_, _, _, _) => self.to_string(),
            IROperandKind::Label(_, _) => self.to_string(),
        }
    }
    

    // ================== IROperand.pool =================== //


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


// ============================================================================== //
//                                op::Instruction
// ============================================================================== //

/// `IRInsn`: IR instruction
#[derive(Debug, Clone, PartialEq)]
pub struct IRInsn {
    pub opc : IROpcode,
    pub opr : Vec<IROperand>,
    pub opb : &'static str,
    pub byt : IRValue,
    pub is_applied : bool,
}

impl IRInsn {

    // Init Opcode POOL
    thread_local! {
        /// Register Map (Shared Global)
        pub static IR_REG_POOL : Rc<RefCell<Vec<Rc<RefCell<IROperand>>>>> = Rc::new(RefCell::new(Vec::new()));
        /// Insn Map (Shared Global)
        pub static IR_INSN_POOL: Rc<RefCell<Vec<Rc<RefCell<IRInsn>>>>> = Rc::new(RefCell::new(Vec::new()));
    }

    /// Get Undef Insn
    pub fn undef() -> IRInsn {
        IRInsn {
            opc: IROpcode::new("insn", Vec::new(), "Undef"),
            opr: Vec::new(),
            opb: "",
            byt: IRValue::u32(0),
            is_applied: false,
        }
    }

    /// Define IRInsn Temp and add to pool
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
        IRInsn::insn_pool_push(insn);
        // return
        IRInsn::insn_pool_nget(name).borrow().clone()
    }

    /// Define Register Temp and add to pool
    pub fn reg(name: &'static str, val: IRValue) -> IROperand {
        let reg = IROperand::reg(name, val);
        Self::reg_pool_push(reg.clone());
        Self::reg_pool_nget(name).borrow().clone()
    }

    /// apply temp to IRInsn
    pub fn apply(name: &'static str, opr: Vec<IROperand>) -> IRInsn {
        let mut insn = IRInsn::insn_pool_nget(name).borrow().clone();
        insn.encode(opr)
    }

    // ================= IRInsn.insn_pool ================== //

    /// delete Insn from pool
    pub fn insn_pool_del(name: &'static str) {
        // Get index
        let idx = Self::insn_pool_idx(name);
        // Delete
        Self::IR_INSN_POOL.with(|pool| pool.borrow_mut().remove(idx));
    }

    /// Get Insn index from pool
    pub fn insn_pool_idx(name: &'static str) -> usize {
        Self::IR_INSN_POOL.with(|pool| pool.borrow().iter().position(|r| r.borrow().name() == name).unwrap())
    }

    /// get Insn from pool
    pub fn insn_pool_nget(name: &'static str) -> Rc<RefCell<IRInsn>> {
        Self::IR_INSN_POOL.with(|pool| pool.borrow().iter().find(|r| r.borrow().name() == name).unwrap().clone())
    }

    /// Set Insn from pool
    pub fn insn_pool_nset(name: &'static str, insn: IRInsn) {
        Self::IR_INSN_POOL.with(|pool| pool.borrow_mut().iter_mut().find(|r| r.borrow().name() == name).unwrap().replace(insn));
    }

    /// Insert Insn from pool
    pub fn insn_pool_push(insn: IRInsn) {
        Self::IR_INSN_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(insn))));
    }

    /// Get Insn from pool
    pub fn insn_pool_get(index: usize) -> Rc<RefCell<IRInsn>> {
        Self::IR_INSN_POOL.with(|pool| pool.borrow()[index].clone())
    }

    /// Set Insn from pool
    pub fn insn_pool_set(index: usize, insn: IRInsn) {
        Self::IR_INSN_POOL.with(|pool| pool.borrow_mut()[index].replace(insn));
    }

    /// Clear Temp Insn Pool
    pub fn insn_pool_clr() {
        Self::IR_INSN_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Get info of Insn Pool
    pub fn insn_pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Instructions (Num = {}):\n", Self::insn_pool_size()));
        
        for i in 0..Self::insn_pool_size() {
            let insn = Self::insn_pool_get(i).borrow().clone();
            info.push_str(&format!("- {}   {} ({})\n", insn.info(), insn.ty(), insn.opb));
        }
        info
    }

    /// Get size of Insn Pool
    pub fn insn_pool_size() -> usize {
        Self::IR_INSN_POOL.with(|pool| pool.borrow().len())
    }

    // ================== IRInsn.reg_pool ================== //

    /// Pool set reg by index
    pub fn reg_pool_set(idx: usize, reg: IROperand) {
        Self::IR_REG_POOL.with(|pool| pool.borrow_mut()[idx] = Rc::new(RefCell::new(reg)));
    }

    /// Pool push reg
    pub fn reg_pool_push(reg: IROperand) {
        Self::IR_REG_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(reg))));
    }

    /// Pool get reg refenence by index
    pub fn reg_pool_get(idx: usize) -> Rc<RefCell<IROperand>> {
        Self::IR_REG_POOL.with(|pool| pool.borrow()[idx].clone())
    }

    /// Pool set reg `index` value by index
    pub fn reg_pool_set_val(idx: usize, val: IRValue) { 
        Self::IR_REG_POOL.with(|pool| pool.borrow_mut()[idx].borrow_mut().set_reg(val));
    }

    /// Pool get reg `index` value by index
    pub fn reg_pool_get_val(idx: usize) -> IRValue {
        Self::IR_REG_POOL.with(|pool| pool.borrow()[idx].borrow().val())
    }

    /// Pool set reg by name
    pub fn reg_pool_nset(name: &'static str , reg: IROperand) {
        Self::IR_REG_POOL.with(|pool| pool.borrow_mut().iter_mut().find(|r| r.borrow().name() == name).unwrap().replace(reg));
    }

    /// Pool get reg refenence by name
    pub fn reg_pool_nget(name: &'static str) -> Rc<RefCell<IROperand>> {
        Self::IR_REG_POOL.with(|pool| pool.borrow().iter().find(|r| r.borrow().name() == name).unwrap().clone())
    }

    /// Pool get reg pool idx by name
    pub fn reg_pool_idx(name: &'static str) -> usize {
        Self::IR_REG_POOL.with(|pool| pool.borrow().iter().position(|r| r.borrow().name() == name).unwrap())
    }

    /// Pool clear
    pub fn reg_pool_clr() {
        Self::IR_REG_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Pool size
    pub fn reg_pool_size() -> usize {
        Self::IR_REG_POOL.with(|pool| pool.borrow().len())
    }

    /// Pool info
    pub fn reg_pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Registers (Num = {}):\n", Self::reg_pool_size()));
        for i in 0..Self::reg_pool_size() {
            let reg = Self::reg_pool_get(i).borrow().clone();
            info.push_str(&format!("- {}\n", reg.info()));
        }
        info
    }



    // ==================== IRInsn.get ===================== //
    
    /// Get byte size of IRInsn
    pub fn size(&self) -> usize {
        self.byt.size()
    }

    /// Name of IRInsn
    pub fn name(&self) -> &'static str {
        self.opc.name()
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


    // ==================== IRInsn.code ==================== //

    /// Get Byte Code of IRInsn
    pub fn code(&self) -> Vec<u8> {
        if !self.is_applied {
            log_error!("Code not applied: {} ", self.opc.name());
        }
        self.byt.val.borrow().clone()
    }

    pub fn funct7(&self) -> u8 {
        self.byt.get_ubyte(0, 25, 7)
    }

    pub fn funct3(&self) -> u8 {
        self.byt.get_ubyte(0, 12, 3)
    }

    pub fn opcode(&self) -> u8 {
        self.byt.get_ubyte(0, 25, 7)
    }

    pub fn rs1(&self) -> u8 {
        self.byt.get_ubyte(0, 15, 5)
    }

    pub fn rs2(&self) -> u8 {
        self.byt.get_ubyte(0, 20, 5)
    }

    pub fn rd(&self) -> u8 {
        self.byt.get_ubyte(0, 7, 5)
    }

    pub fn imm_i(&self) -> u16 {
        self.byt.get_uhalf(0, 20, 12)
    }

    fn set_rs1(&mut self, rs1: u8) {
        println!("rs1: {}", rs1);
        self.byt.set_ubyte(0, 15, 5, rs1);
    }

    fn set_rs2(&mut self, rs2: u8) {
        println!("rs2: {}", rs2);
        self.byt.set_ubyte(0, 20, 5, rs2);
    }

    fn set_rd(&mut self, rd: u8) {
        println!("rd : {}", rd);
        self.byt.set_ubyte(0, 7, 5, rd);
    }

    fn set_imm_i(&mut self, imm: u16) {
        println!("imm: {}", imm);
        self.byt.set_uhalf(0, 20, 12, imm);
    }

    /// Apply IROperand to IROpcode, get new IRInsn
    /// 
    /// ## Byte Code
    /// - Refence: RISC-V ISA Spec.
    /// - Fill bytes according following format:
    /// 
    /// ```txt
    ///  32|31  25|24 20|19 15|14  12|11   7|6  0| bits
    ///    ┌──────┬─────┬─────┬──────┬──────┬────┐
    ///    │  f7  │ rs2 │ rs1 │  f3  │  rd  │ op │  R
    ///    ├──────┴─────┼─────┼──────┼──────┼────┤
    ///    │   imm[12]  │ rs1 │  f3  │  rd  │ op │  I
    ///    ├──────┬─────┼─────┼──────┼──────┼────┤
    ///    │ imm1 │ rs2 │ rs1 │  f3  │ imm0 │ op │  S
    ///    ├─┬────┼─────┼─────┼──────┼────┬─┼────┤
    ///    │3│ i1 │ rs2 │ rs1 │  f3  │ i0 │2│ op │  B
    ///    ├─┴────┴─────┴─────┴──────┼────┴─┼────┤
    ///    │         imm[20]         │  rd  │ op │  U
    ///    ├─┬──────────────┬─┬──────┼──────┼────┤
    ///    │4│     imm1     │3│  i2  │  rd  │ op │  J
    ///    └─┴──────────────┴─┴──────┴──────┴────┘
    /// ```
    pub fn encode(&mut self, opr: Vec<IROperand>) -> IRInsn {
        // Check syms
        if self.check_syms(opr.clone()) {
            // refresh status
            self.opr = opr;
            self.is_applied = true;
            // match opcode type kind and fill bytes by opreands
            match self.opc.kind() {
                IROpcodeKind::I(_, _) => {
                    // rd: u5 -> 7->11
                    let rd = self.opr[0].val().get_byte(0);
                    self.set_rd(rd);

                    // rs1: u5 -> 15->19
                    let rs1 = self.opr[1].val().get_byte(0);
                    self.set_rs1(rs1);

                    // imm: u12 -> 20->32
                    let imm = self.opr[2].val().get_half(0);
                    self.set_imm_i(imm);
                    
                },
                IROpcodeKind::R(_, _) => {
                    // rd: u5 -> 7->11
                    let rd = self.opr[0].val().get_byte(0);
                    self.set_rd(rd);

                    // rs1: u5 -> 15->19
                    let rs1 = self.opr[1].val().get_byte(0);
                    self.set_rs1(rs1);

                    // rs2: u5 -> 20->24
                    let rs2 = self.opr[2].val().get_byte(0);
                    self.set_rs2(rs2);
                },
                IROpcodeKind::J(_, _) => {
                    // TODO: Add IROpcodeKind::J
                },
                _ => {
                    // Do nothing
                },
            }
            self.clone()
        } else {
            // Error
            log_error!("Apply operands failed: {} ", self.opc.name());
            // Revert
            IRInsn::undef()
        }
    }

}


impl fmt::Display for IRInsn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

