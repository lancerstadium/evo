


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::fmt::{self};
use std::rc::Rc;
use std::cell::RefCell;

use crate::log_error;
use crate::util::log::Span;
use crate::ir::op::{IROpcode, IROpcodeKind, IROperand};
use crate::ir::val::IRValue;


// ============================================================================== //
//                                insn::Instruction
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
            opc: IROpcode::new(".insn", Vec::new(), "Undef"),
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
            info.push_str(&format!("- {:<30} {} ({})\n", insn.info(), insn.ty(), insn.opb));
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

    /// Syms of IRInsn
    pub fn syms(&self) -> Vec<i32> {
        self.opc.syms()
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
    pub fn code(&self) -> IRValue {
        if !self.is_applied {
            log_error!("Code not applied: {} ", self.opc.name());
        }
        self.byt.clone()
    }

    pub fn funct7(&self) -> u8 {
        self.byt.get_ubyte(0, 25, 7)
    }

    pub fn funct3(&self) -> u8 {
        self.byt.get_ubyte(0, 12, 3)
    }

    pub fn opcode(&self) -> u8 {
        self.byt.get_ubyte(0, 0, 6)
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

    pub fn imm_s(&self) -> u16 {
        let imm0 = self.byt.get_ubyte(0, 7, 5);
        let imm1 = self.byt.get_ubyte(0, 25, 7);
        let res = (imm0 as u16) | ((imm1 as u16) << 5);
        res
    }

    pub fn imm_b(&self) -> u16 {
        let imm0 = self.byt.get_ubyte(0, 8, 4);
        let imm1 = self.byt.get_ubyte(0, 25, 6);
        let imm2 = self.byt.get_ubyte(0, 7, 1);
        let imm3 = self.byt.get_ubyte(0, 31, 1);
        let res = (imm0 as u16) | ((imm1 as u16) << 4) | ((imm2 as u16) << 11) | ((imm3 as u16) << 12);
        res
    }

    pub fn imm_u(&self) -> u32 {
        self.byt.get_uword(0, 12, 20)
    }

    pub fn imm_j(&self) -> u32 {
        let imm0 = self.byt.get_uhalf(0, 21, 10);
        let imm1 = self.byt.get_ubyte(0, 20, 1);
        let imm2 = self.byt.get_ubyte(0, 12, 8);
        let imm3 = self.byt.get_ubyte(0, 31, 1);
        let res = (imm0 as u32) | ((imm1 as u32) << 11) | ((imm2 as u32) << 19) | ((imm3 as u32) << 20);
        res
    }

    fn set_rs1(&mut self, rs1: u8) {
        self.byt.set_ubyte(0, 15, 5, rs1);
    }

    fn set_rs2(&mut self, rs2: u8) {
        self.byt.set_ubyte(0, 20, 5, rs2);
    }

    fn set_rd(&mut self, rd: u8) {
        self.byt.set_ubyte(0, 7, 5, rd);
    }

    fn set_imm_i(&mut self, imm: u16) {
        self.byt.set_uhalf(0, 20, 12, imm);
    }

    fn set_imm_s(&mut self, imm: u16) {
        let imm0 = (imm & 0x1f) as u8;
        let imm1 = (imm >> 5) as u8;
        self.byt.set_ubyte(0, 7, 5, imm0);
        self.byt.set_ubyte(0, 25, 7, imm1);
    }

    fn set_imm_b(&mut self, imm: u16) {
        let imm0 = (imm & 0xf) as u8;
        let imm1 = ((imm >> 5) & 0x3f) as u8;
        let imm2 = ((imm >> 11) & 0x1) as u8;
        let imm3 = ((imm >> 12) & 0x1) as u8;
        self.byt.set_ubyte(0, 8, 4, imm0);
        self.byt.set_ubyte(0, 25, 6, imm1);
        self.byt.set_ubyte(0, 7, 1, imm2);
        self.byt.set_ubyte(0, 31, 1, imm3);
    }

    fn set_imm_u(&mut self, imm: u32) {
        self.byt.set_uword(0, 12, 20, imm);
    }

    fn set_imm_j(&mut self, imm: u32) {
        let imm0 = (imm & 0x3ff) as u16;
        let imm1 = ((imm >> 11) & 0x1) as u8;
        let imm2 = ((imm >> 19) & 0xff) as u8;
        let imm3 = ((imm >> 20) & 0x1) as u8;
        self.byt.set_uhalf(0, 21, 10, imm0);
        self.byt.set_ubyte(0, 20, 1, imm1);
        self.byt.set_ubyte(0, 12, 8, imm2);
        self.byt.set_ubyte(0, 31, 1, imm3);
    }

    /// is jump instruction
    pub fn is_jump(&self) -> bool {
        match self.name() {
            "jal" | "jalr" => true,
            _ => false
        }
    }

    /// is branch instruction
    pub fn is_branch(&self) -> bool {
        match self.name() {
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => true,
            _ => false
        }
    }

    /// Apply IROperand to IROpcode, get new IRInsn
    /// 
    /// ## Byte Code
    /// - Refence: RISC-V ISA Spec.
    /// - Fill bytes according following format:
    /// 
    /// ```txt
    ///  32|31    25|24 20|19 15|  12|11   7|6  0| bits
    ///    ┌────────┬─────┬─────┬────┬──────┬────┐
    ///    │   f7   │ rs2 │ rs1 │ f3 │  rd  │ op │  R
    ///    ├────────┴─────┼─────┼────┼──────┼────┤
    ///    │    imm[12]   │ rs1 │ f3 │  rd  │ op │  I
    ///    ├────────┬─────┼─────┼────┼──────┼────┤
    ///    │  imm1  │ rs2 │ rs1 │ f3 │ imm0 │ op │  S
    ///    ├─┬──────┼─────┼─────┼────┼────┬─┼────┤
    ///    │3│  i1  │ rs2 │ rs1 │ f3 │ i0 │2│ op │  B
    ///    ├─┴──────┴─────┴─────┴────┼────┴─┼────┤
    ///    │          imm[20]        │  rd  │ op │  U
    ///    ├─┬──────────┬─┬──────────┼──────┼────┤
    ///    │3│   imm0   │1│   imm2   │  rd  │ op │  J
    ///    └─┴──────────┴─┴──────────┴──────┴────┘
    /// ```
    pub fn encode(&mut self, opr: Vec<IROperand>) -> IRInsn {
        if opr.len() == 0 {
            let mut res = self.clone();
            res.is_applied = true;
            return res;
        }
        let mut opr = opr;
        // Check syms
        if self.check_syms(opr.clone()) {
            // match opcode type kind and fill bytes by opreands
            match self.opc.kind() {
                IROpcodeKind::R(_, _) => {
                    // rd: u5 -> 7->11
                    let rd = opr[0].val().get_byte(0);
                    self.set_rd(rd);
                    // rs1: u5 -> 15->19
                    let rs1 = opr[1].val().get_byte(0);
                    self.set_rs1(rs1);
                    // rs2: u5 -> 20->24
                    let rs2 = opr[2].val().get_byte(0);
                    self.set_rs2(rs2);
                },
                IROpcodeKind::I(_, _) => {
                    // rd: u5 -> 7->11
                    let rd = opr[0].val().get_byte(0);
                    self.set_rd(rd);
                    // rs1: u5 -> 15->19
                    let rs1 = opr[1].val().get_byte(0);
                    self.set_rs1(rs1);
                    // imm: u12 -> 20->32
                    let imm = opr[2].val().get_half(0);
                    self.set_imm_i(imm);
                    // refresh imm
                    opr.pop();
                    opr.push(IROperand::imm(IRValue::bit(12, imm as i128)));
                },
                IROpcodeKind::S(_, _) => {
                    // rs2: u5 -> 20->24
                    let rs2 = opr[0].val().get_byte(0);
                    self.set_rs2(rs2);
                    // rs1: u5 -> 15->19
                    let rs1 = opr[1].val().get_byte(0);
                    self.set_rs1(rs1);
                    // imm: S
                    let imm = opr[2].val().get_half(0);
                    self.set_imm_s(imm);
                    // refresh imm
                    opr.pop();
                    opr.push(IROperand::imm(IRValue::bit(12, imm as i128)));
                },
                IROpcodeKind::B(_, _) => {
                    // rs2: u5 -> 20->24
                    let rs2 = opr[0].val().get_byte(0);
                    self.set_rs2(rs2);
                    // rs1: u5 -> 15->19
                    let rs1 = opr[1].val().get_byte(0);
                    self.set_rs1(rs1);
                    // imm: B
                    let imm = opr[2].val().get_half(0);
                    self.set_imm_b(imm);
                    // refresh imm
                    opr.pop();
                    opr.push(IROperand::imm(IRValue::bit(12, imm as i128)));
                },
                IROpcodeKind::U(_, _) => {
                    // rd: u5 -> 7->11
                    let rd = opr[0].val().get_byte(0);
                    self.set_rd(rd);
                    // imm: U
                    let imm = opr[1].val().get_word(0);
                    self.set_imm_u(imm);
                    // refresh imm
                    opr.pop();
                    opr.push(IROperand::imm(IRValue::bit(12, imm as i128)));
                },
                IROpcodeKind::J(_, _) => {
                    // rd: u5 -> 7->11
                    let rd = opr[0].val().get_byte(0);
                    self.set_rd(rd);
                    // imm: J
                    let imm = opr[1].val().get_word(0);
                    self.set_imm_j(imm);
                    // refresh imm
                    opr.pop();
                    opr.push(IROperand::imm(IRValue::bit(20, imm as i128)));
                },
                _ => {
                    // Do nothing
                },
            }
            // refresh status
            let mut res = self.clone();
            res.opr = opr;
            res.is_applied = true;
            res
        } else {
            // Error
            log_error!("Apply operands failed: {} , check syms", self.opc.name());
            // Revert
            IRInsn::undef()
        }
    }


    /// decode from IRValue
    pub fn decode(value: IRValue) -> IRInsn {
        let mut res = IRInsn::undef();
        // 1. check scale
        if value.scale_sum() != 32 {
            log_error!("Invalid insn scale: {}", value.scale_sum());
            return res;
        }
        // 2. decode opc
        res.byt = value;
        let mut opr = vec![];
        match (res.opcode(), res.funct3(), res.funct7()) {
            // 2.1 R-Type
            (0b0110011, f3, f7) => {
                // Get oprands
                // a. rd
                opr.push(IRInsn::reg_pool_get(res.rd() as usize).borrow().clone());
                // b. rs1
                opr.push(IRInsn::reg_pool_get(res.rs1() as usize).borrow().clone());
                // c. rs2
                opr.push(IRInsn::reg_pool_get(res.rs2() as usize).borrow().clone());
                // find insn
                match (f3, f7) {
                    (0b000, 0b0000000) => res = IRInsn::insn_pool_nget("add").borrow().clone(),
                    (0b000, 0b0100000) => res = IRInsn::insn_pool_nget("sub").borrow().clone(),
                    (0b100, 0b0000000) => res = IRInsn::insn_pool_nget("xor").borrow().clone(),
                    (0b110, 0b0000000) => res = IRInsn::insn_pool_nget("or").borrow().clone(),
                    (0b111, 0b0000000) => res = IRInsn::insn_pool_nget("and").borrow().clone(),
                    (0b001, 0b0000000) => res = IRInsn::insn_pool_nget("sll").borrow().clone(),
                    (0b101, 0b0000000) => res = IRInsn::insn_pool_nget("srl").borrow().clone(),
                    (0b101, 0b0100000) => res = IRInsn::insn_pool_nget("sra").borrow().clone(),
                    (0b010, 0b0000000) => res = IRInsn::insn_pool_nget("slt").borrow().clone(),
                    (0b011, 0b0000000) => res = IRInsn::insn_pool_nget("sltu").borrow().clone(),
                    _ => {

                    }
                }
            },
            // 2.2 I-Type
            (0b0010011, f3, 0b0000000) => {
                // Get oprands
                // a. rd
                opr.push(IRInsn::reg_pool_get(res.rd() as usize).borrow().clone());
                // b. rs1
                opr.push(IRInsn::reg_pool_get(res.rs1() as usize).borrow().clone());
                // c. imm
                opr.push(IROperand::imm(IRValue::bit(12, res.imm_i() as i128)));
                // find insn
                match f3 {
                    0b000 => res = IRInsn::insn_pool_nget("addi").borrow().clone(),
                    0b010 => res = IRInsn::insn_pool_nget("slti").borrow().clone(),
                    0b011 => res = IRInsn::insn_pool_nget("sltiu").borrow().clone(),
                    0b100 => res = IRInsn::insn_pool_nget("xori").borrow().clone(),
                    0b110 => res = IRInsn::insn_pool_nget("ori").borrow().clone(),
                    0b111 => res = IRInsn::insn_pool_nget("andi").borrow().clone(),
                    _ => {}
                }
            },
            // 2.3 S-Type
            (0b0100011, f3, 0b0000000) => {
                // Get oprands
                // a. rs1
                opr.push(IRInsn::reg_pool_get(res.rs1() as usize).borrow().clone());
                // b. rs2
                opr.push(IRInsn::reg_pool_get(res.rs2() as usize).borrow().clone());
                // c. imm
                opr.push(IROperand::imm(IRValue::bit(12, res.imm_s() as i128)));
                // find insn
                match f3 {
                    0b000 => res = IRInsn::insn_pool_nget("sb").borrow().clone(),
                    0b001 => res = IRInsn::insn_pool_nget("sh").borrow().clone(),
                    0b010 => res = IRInsn::insn_pool_nget("sw").borrow().clone(),
                    _ => {}
                }
            },
            // 2.4 B-Type
            (0b1100011, f3, 0b0000000) => {
                // Get oprands
                // a. rs1
                opr.push(IRInsn::reg_pool_get(res.rs1() as usize).borrow().clone());
                // b. rs2
                opr.push(IRInsn::reg_pool_get(res.rs2() as usize).borrow().clone());
                // c. imm
                opr.push(IROperand::imm(IRValue::bit(12, res.imm_b() as i128)));
                // find insn
                match f3 {
                    0b000 => res = IRInsn::insn_pool_nget("beq").borrow().clone(),
                    0b001 => res = IRInsn::insn_pool_nget("bne").borrow().clone(),
                    _ => {}
                }
            }
            _ => {

            }

        }
        // 3. encode
        res.encode(opr)
    }



    /// From string to IRInsn
    pub fn from_string(str: &'static str) -> IRInsn {
        // 1. Deal with string
        let str = str.trim();
        // Check if the string has space
        if !str.contains(' ') {
            let name = str;
            let mut res = IRInsn::insn_pool_nget(name).borrow().clone();
            res = res.encode(vec![]);
            return res;
        }
        // 2. Divide in first space and Get Opcode: `[opc] [opr1], [opr2], ...`
        let mut part = str.splitn(2, ' ');
        // 3. Find Insn from pool
        let res = IRInsn::insn_pool_nget(part.next().unwrap().trim());
        // 4. Collect extra part: Divide by `,` and Get Operands str
        let extra_part = part.next().unwrap();
        let opr = extra_part.split(',').collect::<Vec<_>>();
        // 5. Get Operands by sym
        let opr_sym = res.borrow().syms();
        let mut opr_vec = Vec::new();
        for i in 0..opr_sym.len() {
            let mut r = IROperand::from_string(opr_sym[i], opr[i].trim());
            if opr_sym[i] == 0 {
                r = IROperand::imm(IRValue::u32(r.val().get_word(0)));
            }
            opr_vec.push(r);
        }
        // 5. Encode IRInsn
        let res = res.borrow_mut().encode(opr_vec);
        res
    }
}


impl fmt::Display for IRInsn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}




// ============================================================================== //
//                                insn::IRBasicBlock
// ============================================================================== //


/// IRBasicBlock
pub struct IRBasicBlock {
    
    pub src_insns: Vec<IRInsn>,
    pub ir_insns: Vec<IRInsn>,
    pub trg_insns: Vec<IRInsn>,
    pub liveness_regs: Vec<usize>,

    /// If the block is lifted to EVO ir arch: src -> ir
    pub is_lifted: bool,
    /// If the block is lowered to target arch: ir -> trg
    pub is_lowered: bool,
}


impl IRBasicBlock {


    pub fn init(src_insns: Vec<IRInsn>) -> IRBasicBlock {
        Self {
            src_insns,
            ir_insns: vec![],
            trg_insns: vec![],
            liveness_regs: vec![],
            is_lifted: false,
            is_lowered: false
        }
    }

}