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
use crate::{log_error, log_warning};
use crate::util::log::Span;

use super::ty::IRTypeKind;



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


    /// Set Reg value
    pub fn set_reg(&mut self, val: IRValue) {
        match self {
            IROperandKind::Reg(_, _) => *self = IROperandKind::Reg(self.name(), val),
            _ => *self = IROperandKind::Reg("<Und>", val),
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
        self.0.borrow_mut().set_reg(val);
    }

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
    fn insn_init(&mut self);

    /// Opcode Info: `OpcodeName OpcodeValue, OpcodeValue ...` String
    fn insn_info(&self) -> String;

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
    insn_map: RefCell<Vec<IRInsn>>

}

impl IRArch {

    /// Create new `IRArch`
    pub fn new() -> Self {
        let mut arch = Self {
            reg_map: RefCell::new(Vec::new()),
            insn_map: RefCell::new(Vec::new()),
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

    

    /// 1. Insn temp Map Init
    fn insn_init(&mut self) {
        self.insn_map = RefCell::new(vec![
            // RISCV Instruction Format:            32|31  25|24 20|19 15|  |11  7|6    0|
            // Type: R         [rd, rs1, rs2]         |  f7  | rs2 | rs1 |f3|  rd |  op  |
            IRInsn::def("add" , vec![1, 1, 1], "R", "0b0000000. ........ .000.... .0110011"),
            IRInsn::def("sub" , vec![1, 1, 1], "R", "0b0100000. ........ .000.... .0110011"),
            IRInsn::def("or"  , vec![1, 1, 1], "R", "0b0000000. ........ .111.... .0110011"),
            IRInsn::def("xor" , vec![1, 1, 1], "R", "0b0000000. ........ .100.... .0110011"),
            IRInsn::def("sll" , vec![1, 1, 1], "R", "0b0000000. ........ .001.... .0110011"),
            IRInsn::def("srl" , vec![1, 1, 1], "R", "0b0000000. ........ .101.... .0110011"),
            IRInsn::def("sra" , vec![1, 1, 1], "R", "0b0100000. ........ .101.... .0110011"),
            IRInsn::def("slt" , vec![1, 1, 1], "R", "0b0000000. ........ .010.... .0110011"),
            IRInsn::def("sltu", vec![1, 1, 1], "R", "0b0000000. ........ .011.... .0110011"),
            // Type: I         [rd, rs1, imm]         |    imm     | rs1 |f3|  rd |  op  |
            IRInsn::def("addi", vec![1, 1, 0], "I", "0b0000000. ........ .000.... .0010011"),
        ]);
    }
    
    /// 2. Opcode Info
    fn insn_info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Instructions (Num = {}):\n", self.insn_map.borrow().len()));
        for insn in self.insn_map.borrow().iter() {
            info.push_str(&format!("- {}   {} ({})\n", insn.info(), insn.ty(), insn.bin(0, -1, false)));
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
        arch.insn_init();
        println!("{}", arch.insn_info());
        let insn1 = IRInsn::apply("xor", vec![IROperand::reg("eax", IRValue::u8(8)), IROperand::reg("eax", IRValue::u8(8)), IROperand::reg("ebx", IRValue::u8(9))]);
        println!("{}", insn1);
    }
}