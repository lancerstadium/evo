//! `evo::ir::ctx` : IR Context
//! 
//!

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::{log_error, log_warning};
use crate::util::log::Span;
use crate::ir::val::IRValue;
use crate::ir::ty::IRType;
use crate::ir::op::{IROperand, IRInsn};


// ============================================================================== //
//                                ctx::ArchInfo
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
    /// Base of Addr: 0x04000000
    const BASE_ADDR: usize;
    /// Mem size: default 64MB = 4 * 1024 * 1024
    const MEM_SIZE: usize;
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
    /// Get Register index
    fn get_reg_idx(&self, name: &'static str) -> IRValue;
    /// Set Register index
    fn set_reg_idx(&mut self, name: &'static str, value: IRValue);
    /// Get refence of Register
    fn reg(&mut self, name: &'static str) -> RefCell<IROperand>;
    /// Get Register Value
    fn reg_read(&self, name: &'static str) -> IRValue;
    /// Set Register Value
    fn reg_write(&mut self, name: &'static str, value: IRValue);


    // ===================== Opcode ===================== //

    /// Opcode Map Init
    fn insn_init(&mut self);
    /// Opcode Info: `OpcodeName OpcodeValue, OpcodeValue ...` String
    fn insn_info(&self) -> String;


    // ===================== Memory ===================== //

    /// Mem Init
    fn seg_init(&mut self);
    /// Check Mem Bound
    fn seg_bound(&self, addr: IRValue) -> bool;

}




// ============================================================================== //
//                              ctx::IRContext
// ============================================================================== //


/// `IRContext`: Context of the `evo-ir` architecture
/// 
/// ### Process and Thread
/// 
/// - Here are the process and thread in the `evo-ir` architecture
/// 
/// ```
///           ┌──────────────┬────────────────┬──────┐
///  Process  │ Code Segment │  Data Segment  │ Heap │
///  ───────  ├──────────────┴──┬─────────────┴──────┤
///  Threads  │      Stack      │     Registers      │
///           └─────────────────┴────────────────────┘
///
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IRContext {
    /// `reg_map`: Register Map (Shared Local Process)
    reg_map: Rc<RefCell<Vec<RefCell<IROperand>>>>,
    /// `opcode_map`: Opcode Map (Shared Local Process)
    insn_map: Rc<RefCell<Vec<IRInsn>>>,

    /// `segments`: Runtime Memory Process: Code Segment, Data Segment. (Local Process)
    segments: RefCell<Vec<IRValue>>,
    /// `heap`: Runtime Memory Process: Heap. (Local Process)
    heap: RefCell<Vec<IRValue>>,

    /// `registers`: Register File, Store register value (Local Thread)
    registers: RefCell<Vec<IRValue>>,
    /// `stack`: Stack, Store stack value (Local Thread)
    stack: RefCell<Vec<IRValue>>,
}

impl IRContext {

    /// Init new `IRContext`
    pub fn init() -> Self {
        let mut arch = Self {
            reg_map: Rc::new(RefCell::new(Vec::new())),
            insn_map: Rc::new(RefCell::new(Vec::new())),

            segments: RefCell::new(Vec::new()),
            heap: RefCell::new(Vec::new()),

            registers: RefCell::new(Vec::new()),
            stack: RefCell::new(Vec::new()),
        };
        arch.reg_init();
        arch.insn_init();
        arch.seg_init();
        arch
    }


    /// Get segment size
    pub fn seg_size(&self) -> usize {
        self.segments.borrow().len()
    }

    /// Get segment scale
    pub fn seg_scale(&self) -> usize {
        self.segments.borrow().iter().map(|x| x.scale_sum()).sum()
    }
    
}


impl Default for IRContext {
    /// Set default function for `IRContext`.
    fn default() -> Self {
        Self::init()
    }
}


impl ArchInfo for IRContext {


    // =================== IRCtx.const ===================== //

    // 1. Set Constants
    const NAME: &'static str = "evo32";
    const BYTE_SIZE: usize = 1;
    const ADDR_SIZE: usize = 32;
    const WORD_SIZE: usize = 32;
    const FLOAT_SIZE: usize = 32;
    const BASE_ADDR: usize = 0x04000000;
    const MEM_SIZE: usize = 4 * 1024 * 1024;
    const REG_NUM: usize = 32;


    // =================== IRCtx.info ====================== //

    /// 2. Get Arch string
    fn to_string () -> String {
        format!("{}", Self::NAME)
    }

    /// 3. Get ArchInfo string
    fn info() -> String {
        format!("Arch Info: \n- Name: {}\n- Byte Size: {}\n- Addr Size: {}\n- Word Size: {}\n- Float Size: {}\n- Base Addr: 0x{:x}\n- Mem Size: {}\n- Reg Num: {}", 
            Self::NAME, Self::BYTE_SIZE, Self::ADDR_SIZE, Self::WORD_SIZE, Self::FLOAT_SIZE, Self::BASE_ADDR, Self::MEM_SIZE, Self::REG_NUM)
    }

    /// 3. Get Name
    fn name() -> &'static str {
        Self::NAME
    }


    // =================== IRCtx.reg ======================= //

    /// 4. Register Map Init
    fn reg_init(&mut self) {
        // 1. Init reg name and index
        self.reg_map = Rc::new(RefCell::new(vec![
            RefCell::new(IROperand::reg("x0", IRValue::u5(0))),
            RefCell::new(IROperand::reg("x1", IRValue::u5(1))),
            RefCell::new(IROperand::reg("x2", IRValue::u5(2))),
            RefCell::new(IROperand::reg("x3", IRValue::u5(3))),
            RefCell::new(IROperand::reg("x4", IRValue::u5(4))),
            RefCell::new(IROperand::reg("x5", IRValue::u5(5))),
            RefCell::new(IROperand::reg("x6", IRValue::u5(6))),
            RefCell::new(IROperand::reg("x7", IRValue::u5(7))),
            RefCell::new(IROperand::reg("x8", IRValue::u5(8))),
        ]));
        // 2. Init reg file value: [0, 0, ...] * REG_NUM (u32)
        for _ in 0..Self::REG_NUM {
            self.registers.borrow_mut().push(IRValue::u32(0));
        }
        // 3. Check register map num == REG_NUM
        if Self::REG_NUM != self.reg_map.borrow().len() {
            log_warning!("Register map not match with address size: {} != {}", self.reg_map.borrow().len() , Self::REG_NUM);
        }
    }

    /// 5. Reg Info
    fn reg_info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Registers (Num = {}):\n", self.reg_map.borrow().len()));
        for reg in self.reg_map.borrow().iter() {
            let idx_str = reg.borrow().val().bin_scale(0, -1, false);
            info.push_str(&format!("- {:<9} ({:>2}: {}) -> {}\n", reg.borrow().to_string(), reg.borrow().val().to_string(), idx_str, self.reg_read(reg.borrow().name()).bin(0, -1, false)));
        }
        info
    }

    /// 6. Get Register index
    fn get_reg_idx(&self, name: &'static str) -> IRValue {
        self.reg_map.borrow().iter().find(|reg| reg.borrow().name() == name).unwrap().borrow().val()
    }
    /// 7. Set Register index
    fn set_reg_idx(&mut self, name: &'static str, value: IRValue) {
        // Set value according to name and value
        self.reg_map.borrow().iter().find(|reg| reg.borrow().name() == name).unwrap().borrow_mut().set_reg(value);
    }

    /// 8. Get Register reference
    fn reg(&mut self, name: &'static str) -> RefCell<IROperand> {
        self.reg_map.borrow().iter().find(|reg| reg.borrow().name() == name).unwrap().clone()
    }

    /// 9. Read Register Value (u32)
    fn reg_read(&self, name: &'static str) -> IRValue {
        let idx = self.get_reg_idx(name).get_byte(0) as usize;
        let v = self.registers.borrow()[idx].clone();
        v
    }

    /// 10. Write Register Value (u32)
    fn reg_write(&mut self, name: &'static str, value: IRValue) {
        let idx = self.get_reg_idx(name).get_byte(0) as usize;
        self.registers.borrow_mut()[idx].set_val(value);
    }

    
    // =================== IRCtx.insn ====================== //

    /// 1. Insn temp Map Init
    fn insn_init(&mut self) {
        self.insn_map = Rc::new(RefCell::new(vec![
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
        ]));
    }
    
    /// 2. Opcode Info
    fn insn_info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Instructions (Num = {}):\n", self.insn_map.borrow().len()));
        for insn in self.insn_map.borrow().iter() {
            info.push_str(&format!("- {}   {} ({})\n", insn.info(), insn.ty(), insn.opbit()));
        }
        info
    }


    // =================== IRCtx.seg ======================= //

    /// 1. Segment Mem Init
    fn seg_init(&mut self) {
        // Mem Space Init: 1 Mem Space
        self.segments.borrow_mut().push(IRValue::new(IRType::array(IRType::u8(), Self::MEM_SIZE / 8)));
    }

    /// 2. Segment Mem bound check (addr: u32)
    fn seg_bound(&self, addr: IRValue) -> bool {
        let index = addr.get_u32(0) as usize;
        let is_valid = index >= Self::BASE_ADDR && index < Self::BASE_ADDR + Self::MEM_SIZE;
        if !is_valid {
            log_error!("Memory access out of bounds: 0x{:x}", index);
        }
        is_valid
    }

}







// ============================================================================== //
//                                Unit Tests
// ============================================================================== //

#[cfg(test)]
mod ctx_test {

    use super::*;


    #[test]
    fn insn_info() {
        IRContext::init();
        let insn1 = IRInsn::apply(
            "xor", vec![
                IROperand::reg("x1", IRValue::u5(8)), 
                IROperand::reg("x2", IRValue::u5(8)), 
                IROperand::reg("x3", IRValue::u5(9))
            ]
        );
        println!("{}", insn1);
    }

    #[test]
    fn ctx_info() {
        println!("{}", IRContext::info());

        let mut ctx = IRContext::init();
        ctx.set_reg_idx("x3", IRValue::u5(9));
        assert_eq!(ctx.get_reg_idx("x3"), IRValue::u5(9));
        
        println!("{}", ctx.insn_info());

        ctx.set_reg_idx("x4", IRValue::u5(13));
        assert_eq!(ctx.get_reg_idx("x4"), IRValue::u5(13));
        ctx.reg("x4").borrow_mut().set_reg(IRValue::u5(9));
        assert_eq!(ctx.get_reg_idx("x4"), IRValue::u5(9));
        ctx.set_reg_idx("x4", IRValue::u5(13));
        assert_eq!(ctx.get_reg_idx("x4"), IRValue::u5(13));

        ctx.reg_write("x6", IRValue::u32(65535));
        assert_eq!(ctx.reg_read("x6").bin(0, 1, false), "0b11111111");
        println!("{}", ctx.reg_info());

        let ctx2 = IRContext::init();
        // Compare Registers
        assert_ne!(ctx, ctx2);


        println!("mem size: {}", ctx.seg_size());
        println!("mem scale: {}", ctx.seg_scale());
    }


}