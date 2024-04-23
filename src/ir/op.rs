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

use crate::ir::val::IRValue;
use crate::log_warning;
use crate::util::log::Span;

use super::ty::IRTypeKind;



// ============================================================================== //
//                              op::IROperandKind
// ============================================================================== //

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROperandKind {
    /// |  val  |
    Imm(IRValue),

    /// |  name*  |  val   |
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
            IROperandKind::Imm(val) => val.hex(0, -1),
            // Get Reg value's hex
            IROperandKind::Reg(_, val) => val.hex(0, -1),
            IROperandKind::Mem(_, _, _, _) => self.val().hex(0, -1),
            // Get Label ptr's hex
            IROperandKind::Label(_, val) => val.hex(0, -1),
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

    /// Get String of the operand
    pub fn to_string(&self) -> String {
        match self {
            IROperandKind::Imm(val) =>  format!("<Imm>: {}", val.kind()),
            IROperandKind::Reg(name, val) => format!("{}: {}", name, val.kind()),
            IROperandKind::Mem(_, _, _, _) => format!("<Mem> : {}", self.val().kind()),
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
    pub fn dft_reg() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Reg("<Reg>", IRValue::u32(0)))))
    }

    /// New default Imm
    pub fn dft_imm() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Imm(IRValue::u32(0)))))
    }

    /// New default Mem
    pub fn dft_mem() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Mem(IRValue::u32(0), IRValue::u32(0), IRValue::u32(0), IRValue::u32(0)))))
    }

    /// New default Label
    pub fn dft_label() -> Self {
        Self(Rc::new(RefCell::new(IROperandKind::Label("<Label>", IRValue::u32(0)))))
    }


    // ================== IROperand.get ==================== //

    /// Get Operand value
    pub fn val(&self) -> IRValue {
        self.0.borrow().val()
    }

    /// Get Operand name
    pub fn name(&self) -> &'static str {
        self.0.borrow().name()
    }

    // ================== IROperand.set ==================== //

    /// Set Operand value
    pub fn set_reg(&mut self, val: IRValue) {
        let mut kind = self.0.borrow_mut();
        *kind = IROperandKind::Reg(kind.name(), val);
    }

    /// To String
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
    /// Special opcode: [opcode]
    Special(&'static str),

    /// Unary operand opcode: [opcode] [operand]
    Unary(&'static str, RefCell<IROperand>),

    /// Binary operand opcode: [opcode] [operand1], [operand2]
    Binary(&'static str, RefCell<IROperand>, RefCell<IROperand>),

    /// Ternary operand opcode: [opcode] [operand1], [operand2], [operand3]
    Ternary(&'static str, RefCell<IROperand>, RefCell<IROperand>, RefCell<IROperand>),

    /// Quaternary operand opcode: [opcode] [operand1], [operand2], [operand3], [operand4]
    Quaternary(&'static str, RefCell<IROperand>, RefCell<IROperand>, RefCell<IROperand>, RefCell<IROperand>),

}


impl IROpcodeKind {

    /// Get Name of OpcodeKind
    pub fn name(&self) -> &'static str {
        match self {
            IROpcodeKind::Special(name) => name,
            IROpcodeKind::Unary(name, _) => name,
            IROpcodeKind::Binary(name, _, _) => name,
            IROpcodeKind::Ternary(name, _, _, _) => name,
            IROpcodeKind::Quaternary(name, _, _, _, _) => name,
        }
    }

    /// To string
    pub fn to_string(&self) -> String {
        match self {
            IROpcodeKind::Special(name) => name.to_string(),
            IROpcodeKind::Unary(name, ope) => format!("{:<6} {}", name, ope.borrow().to_string()),
            IROpcodeKind::Binary(name, ope1, ope2) => format!("{:<6} {}, {}", name, ope1.borrow().to_string(), ope2.borrow().to_string()),
            IROpcodeKind::Ternary(name, ope1, ope2, ope3) => format!("{:<6} {}, {}, {}", name, ope1.borrow().to_string(), ope2.borrow().to_string(), ope3.borrow().to_string()),
            IROpcodeKind::Quaternary(name, ope1, ope2, ope3, ope4) => format!("{:<6} {}, {}, {}, {}", name, ope1.borrow().to_string(), ope2.borrow().to_string(), ope3.borrow().to_string(), ope4.borrow().to_string()),
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
pub struct IROpcode(Rc<IROpcodeKind>);

impl IROpcode {

    // Init opcode pool
    thread_local! {
        /// Opcode Pool
        static OPCODE_POOL: RefCell<HashMap<&'static str, IROpcode>> = RefCell::new(HashMap::new());
    }

    // =================== IROpcode.get ==================== //

    /// Regist Opcodekind to pool and return Reference of IROpcode
    pub fn regist(kind: IROpcodeKind) -> IROpcode {
        IROpcode::OPCODE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let name = kind.name();
            pool.insert(name, IROpcode(Rc::new(kind)));
            pool.get(name).unwrap().clone()
        })
    }

    /// Get Special Opcode
    pub fn special(name: &'static str) -> Self {
        Self::regist(IROpcodeKind::Special(name))
    }

    /// Get Unary Opcode
    pub fn unary(name: &'static str, operand: RefCell<IROperand>) -> Self {
        Self::regist(IROpcodeKind::Unary(name, operand))
    }

    /// Get Binary Opcode
    pub fn binary(name: &'static str, operand1: RefCell<IROperand>, operand2: RefCell<IROperand>) -> Self {
        Self::regist(IROpcodeKind::Binary(name, operand1, operand2))
    }

    /// Get Ternary Opcode
    pub fn ternary(name: &'static str, operand1: RefCell<IROperand>, operand2: RefCell<IROperand>, operand3: RefCell<IROperand>) -> Self {
        Self::regist(IROpcodeKind::Ternary(name, operand1, operand2, operand3))
    }
    
    /// Get Quaternary Opcode
    pub fn quaternary(name: &'static str, operand1: RefCell<IROperand>, operand2: RefCell<IROperand>, operand3: RefCell<IROperand>, operand4: RefCell<IROperand>) -> Self {
        Self::regist(IROpcodeKind::Quaternary(name, operand1, operand2, operand3, operand4))
    }

    /// Get Name of Opcode
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Get a refence of OpcodeKind
    pub fn kind(&self) -> &IROpcodeKind {
        &self.0
    }

    /// Find Opcode by name
    pub fn find(name: &'static str) -> Option<IROpcode> {
        IROpcode::OPCODE_POOL.with(|pool| {
            let pool = pool.borrow();
            pool.get(name).map(|op| op.clone())
        })
    }


    // =================== IROpcode.set ==================== //

    
}

impl fmt::Display for IROpcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
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
pub struct IRInsn(Rc<IROpcode>);


impl fmt::Display for IRInsn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================== //
//                                op::ArchInfo
// ============================================================================== //


/// `ArchInfo`: Config information of the architecture
pub trait ArchInfo {


    // ====================== Const ====================== //

    /// Arch name: like "evo"
    const NAME: &'static str;

    /// Number of bytes in a byte: *1*, 2, 4
    const BYTE_SIZE: usize;

    /// Number of bytes in a addr(ptr/reg.size: 0x00 ~ 2^ADDR_SIZE): 8, 16, *32*, 64
    const ADDR_SIZE: usize;

    /// Number of bytes in a word(interger): 8, 16, *32*, 64
    const WORD_SIZE: usize;

    /// Number of bytes in a (float): *32*, 64
    const FLOAT_SIZE: usize;

    /// Number of Registers: 8, 16, *32*, 64
    const REG_NUM: usize;

    /// Get Arch string
    fn to_string () -> String;

    /// Get Info String
    fn info() -> String;

    // ====================== Reg ======================== //

    /// Get Name
    fn name() -> &'static str;

    /// Register Map Init
    fn reg_init(&mut self);

    /// Reg Info: `RegName: RegValue` String
    fn reg_info(&self) -> String;

    /// Get Register
    fn get_reg_val(&self, name: &'static str) -> IRValue;

    /// Set Register
    fn set_reg_val(&mut self, name: &'static str, value: IRValue);

    /// Get refence of Register
    fn reg(&mut self, name: &'static str) -> RefCell<IROperand>;


    // ===================== Opcode ===================== //

    /// Opcode Map Init
    fn opcode_init(&mut self);

    /// Opcode Info: `OpcodeName OpcodeValue, OpcodeValue ...` String
    fn opcode_info(&self) -> String;

}




// ============================================================================== //
//                              op::IRArch
// ============================================================================== //


/// `IRArch`: Config of the `evo-ir` architecture
#[derive(Debug, Clone, PartialEq)]
pub struct IRArch {
    /// `reg_map`: Register Map
    reg_map: RefCell<Vec<RefCell<IROperand>>>,
    /// `opcode_map`: Opcode Map
    opcode_map: RefCell<Vec<IROpcode>>

}

impl IRArch {

    /// Create new `IRArch`
    pub fn new() -> Self {
        let mut arch = Self {
            reg_map: RefCell::new(Vec::new()),
            opcode_map: RefCell::new(Vec::new()),
        };
        arch.reg_init();
        arch
    }
    
}


impl Default for IRArch {
    /// Set default function for `IRArch`.
    fn default() -> Self {
        Self::new()
    }
}


impl ArchInfo for IRArch {

    // 1. Set Constants
    const NAME: &'static str = "evo";
    const BYTE_SIZE: usize = 1;
    const ADDR_SIZE: usize = 32;
    const WORD_SIZE: usize = 32;
    const FLOAT_SIZE: usize = 32;
    const REG_NUM: usize = 32;

    /// 2. Get Arch string
    fn to_string () -> String {
        // Append '-' with the address size
        format!("{}-{}", Self::NAME, Self::ADDR_SIZE)
    }

    /// 3. Get ArchInfo string
    fn info() -> String {
        format!("[{}]:\n - byte: {}\n - addr: {}\n - word: {}\n - float: {}\n - reg: {}", Self::to_string(), Self::BYTE_SIZE, Self::ADDR_SIZE, Self::WORD_SIZE, Self::FLOAT_SIZE, Self::REG_NUM)
    }


    /// 3. Get Name
    fn name() -> &'static str {
        Self::NAME
    }

    /// 4. Register Map Init
    fn reg_init(&mut self) {
        self.reg_map = RefCell::new(vec![
            RefCell::new(IROperand::reg("eax", IRValue::u32(0))),
            RefCell::new(IROperand::reg("ebx", IRValue::u32(0))),
            RefCell::new(IROperand::reg("ecx", IRValue::u32(0))),
            RefCell::new(IROperand::reg("edx", IRValue::u32(0))),
            RefCell::new(IROperand::reg("esi", IRValue::u32(0))),
            RefCell::new(IROperand::reg("edi", IRValue::u32(0))),
            RefCell::new(IROperand::reg("esp", IRValue::u32(0))),
            RefCell::new(IROperand::reg("ebp", IRValue::u32(0))),
        ]);
        if Self::ADDR_SIZE != self.reg_map.borrow().len() {
            log_warning!("Register map not match with address size: {} != {}", self.reg_map.borrow().len() , Self::ADDR_SIZE);
        }
    }

    /// 5. Reg Info
    fn reg_info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Registers (Num = {}):\n", self.reg_map.borrow().len()));
        for reg in self.reg_map.borrow().iter() {
            info.push_str(&format!("- {:<10} ({:>4} : {})\n", reg.borrow().to_string(), reg.borrow().val(), reg.borrow().val().bin(0, -1, false)));
        }
        info
    }

    /// 6. Get Register value
    fn get_reg_val(&self, name: &'static str) -> IRValue {
        self.reg_map.borrow().iter().find(|reg| reg.borrow().name() == name).unwrap().borrow().val()
    }
    /// 7. Set Register value
    fn set_reg_val(&mut self, name: &'static str, value: IRValue) {
        // Set value according to name and value
        self.reg_map.borrow().iter().find(|reg| reg.borrow().name() == name).unwrap().borrow_mut().set_reg(value);
    }

    /// 8. Get Register reference
    fn reg(&mut self, name: &'static str) -> RefCell<IROperand> {
        self.reg_map.borrow().iter().find(|reg| reg.borrow().name() == name).unwrap().clone()
    }

    

    /// 1. Opcode Map Init
    fn opcode_init(&mut self) {

        self.opcode_map = RefCell::new(vec![
            IROpcode::special("nop"),
            IROpcode::special("ret"),
            IROpcode::binary("add", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("sub", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("mul", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("div", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("mod", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("and", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("or" , RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("xor", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
            IROpcode::binary("shl", RefCell::new(IROperand::dft_reg()), RefCell::new(IROperand::dft_reg())),
        ]);
        
    }
    
    /// 2. Opcode Info
    fn opcode_info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Opcode Info (Num = {}):\n", self.opcode_map.borrow().len()));
        let mut idx = 0;
        for opcode in self.opcode_map.borrow().iter() {
            info.push_str(&format!("[{:>3}] {}\n", idx, opcode.to_string()));
            idx += 1;
        }
        info
    }
}







// ============================================================================== //
//                                Unit Tests
// ============================================================================== //

#[cfg(test)]
mod op_test {

    use super::*;

    #[test]
    fn arch_reg() {
        println!("{}", IRArch::info());
        let mut arch = IRArch::new();
        
        arch.set_reg_val("ebx", IRValue::u32(9));
        arch.set_reg_val("eax", IRValue::u32(8));
        assert_eq!(arch.get_reg_val("ebx"), IRValue::u32(9));
        println!("{}", arch.reg_info());


        arch.set_reg_val("ebx", IRValue::u32(13));
        assert_eq!(arch.get_reg_val("ebx"), IRValue::u32(13));
        arch.reg("ebx").borrow_mut().set_reg(IRValue::u32(9));
        assert_eq!(arch.get_reg_val("ebx"), IRValue::u32(9));
        arch.set_reg_val("ebx", IRValue::u32(13));
        assert_eq!(arch.get_reg_val("ebx"), IRValue::u32(13));

        let arch2 = IRArch::new();
        // Compare Registers
        assert_ne!(arch, arch2);
    }

    #[test]
    fn arch_opcode() {
        let mut arch = IRArch::new();
        arch.opcode_init();
        println!("{}", arch.opcode_info());
        println!("{}", IROpcode::find("add").unwrap().to_string());
    }
}