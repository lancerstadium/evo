
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use crate::arch::evo::def::EVO_ARCH;
use crate::arch::info::Arch;
use crate::core::insn::Instruction;
use crate::core::val::Value;



// ============================================================================== //
//                                blk::BasicBlock
// ============================================================================== //


/// BasicBlock
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    

    /// `BasicBlock`: Basic Block Flag
    /// ```txt
    /// (0-7):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ ├─┴─┘   (0-2) 000 is 8-bit , 001 is 16-bit, 010 is 32-bit , 011 is 64-bit
    /// │ │ │ │ │ └────── (0-2) 100 is 80-bit, 101 is 96-bit, 110 is 128-bit, 111 is 256-bit
    /// │ │ │ │ └──────── (3) 0 is little-endian, 1 is big-endian
    /// │ │ │ │         ┌ (4-7) 0000: COND_NO , 0001: COND_ALL, 0010: COND_EQ , 0011: COND_NE
    /// │ │ │ │         │ (4-7) 0100: COND_LT , 0101: COND_GE , 0110: COND_LE , 0111: COND_GT
    /// ├─┴─┴─┘         │ (4-7) 1000: COND_LTU, 1001: COND_GEU, 1010: COND_LEU, 1011: COND_GTU
    /// └───────────────┴ (4-7) 
    /// (8-15):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ │ │ └── (8)  0: is not branch , 1: is branch
    /// │ │ │ │ │ │ └──── (9)  0: is not jump   , 1: is jump
    /// │ │ │ │ │ └────── (10) 0: is not exit   , 1: is exit
    /// │ │ │ │ └──────── (11) 0: is not bb end , 1: is bb end
    /// │ │ │ └────────── <Reserved>
    /// │ │ └──────────── <Reserved>
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// ```
    pub flag: u16,

    pub addr: Option<Value>,
    pub label: Option<String>,

    pub arch: &'static Arch,
    pub insns: Vec<Instruction>,

    pub predecessors: Vec<*mut BasicBlock>,     // Predecessors <- Self
    pub successors: Vec<*mut BasicBlock>,       // Self <- Successors

}


impl BasicBlock {


    // ================== BasicBlock: crl =================== //

    pub fn new(arch: &'static Arch) -> Box<BasicBlock> {
        Box::new(Self {
            flag: 0,
            addr: None,
            label: None,
            predecessors: Vec::new(),
            successors: Vec::new(),
            insns: Vec::new(),
            arch
        })
    }

    pub fn init(insns: Vec<Instruction>) -> Box<BasicBlock> {
        let arch;
        let label;
        if insns.len() == 0 { 
            arch = &EVO_ARCH;
            label = None;
        } else {
            arch = insns[0].arch;
            label = insns[0].label.clone();
        }
        Box::new(Self {
            flag: 0,
            addr: None,
            label,
            predecessors: Vec::new(),
            successors: Vec::new(),
            insns,
            arch
        })
        
    }

    /// Returns the number of instructions in the basic block.
    pub fn size(&self) -> usize {
        self.insns.len()
    }

    pub fn push(&mut self, insn: Instruction) {
        self.insns.push(insn);
    }

    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("{}\n", self.insns.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("\n")));
        info
    }

    // ================== BasicBlock: set =================== //

    pub fn set_addr(&mut self, addr: Value) {
        self.addr = Some(addr);
    }

    pub fn set_label(&mut self, label: String) {
        self.label = Some(label);
    }

    // ================== BasicBlock: get =================== //



    // ================== BasicBlock: jmp =================== //

    pub fn set_branch(&mut self, _then: &mut BasicBlock, _else: &mut BasicBlock) {
        if _then.addr == _else.addr {       // 1 branch
            _then.predecessors.push(self);
            self.successors.push(_then);
        } else {                            // 2 branch
            _then.predecessors.push(self);
            self.successors.push(_then);
            _else.predecessors.push(self);
            self.successors.push(_else);
        }
    }


    // ================== BasicBlock: str =================== //

    pub fn from_string(arch: &'static Arch, s: &str) -> Box<BasicBlock> {
        // Divide insns by "\n"
        let insns = s.split("\n").map(|x| Instruction::from_string(arch, x)).collect::<Vec<Instruction>>();
        Self::init(insns)
    }

}


#[cfg(test)]
mod blk_test {

    use super::*;
    use crate::arch::riscv::def::RISCV32_ARCH;
    use crate::arch::evo::def::EVO_ARCH;
    use crate::core::cpu::CPUState;

    #[test]
    fn blk_init() {
        let cpu = CPUState::init(&RISCV32_ARCH, &EVO_ARCH, None, None, None);

        let mut bb1 = BasicBlock::init(vec![
            Instruction::from_string(&RISCV32_ARCH, "BB1: add x0, x1, x2"),
            Instruction::from_string(&RISCV32_ARCH, "sub x0, x1, x2")
        ]);
        let mut bb2 = BasicBlock::from_string(&RISCV32_ARCH, 
            "BB2: add x0, x1, x2
             sub x0, x1, x2
             lbu x0, x1, 0b11001111"
        );
        let tb1 = cpu.lift(&mut bb1);
        let tb2 = cpu.lift(&mut bb2);
        println!("{}", tb1.info());
        println!("{}", tb2.info());
    }
}