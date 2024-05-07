
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::fmt::{self};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


// ============================================================================== //
//                               info::ArchModeKind
// ============================================================================== //

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ArchModeKind {

    /// Little-endian 16-bit
    L16 = 0b000,
    /// Little-endian 32-bit
    L32 = 0b001,
    /// Little-endian 64-bit
    L64 = 0b010,
    /// Little-endian 128-bit
    L128 = 0b011,

    /// Big-endian 16-bit
    B16 = 0b100,
    /// Big-endian 32-bit
    B32 = 0b101,
    /// Big-endian 64-bit
    B64 = 0b110,
    /// Big-endian 128-bit
    B128 = 0b111,


}



// ============================================================================== //
//                                info::ArchMode
// ============================================================================== //

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ArchMode {
    /// ### Arch Mode flag
    /// ```txt
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ │ ├─┘
    /// │ │ │ │ │ │ └──── (0-1) 00 is 16-bit, 01 is 32-bit, 10 is 64-bit, 11 is 128-bit
    /// │ │ │ │ │ └────── (2) 0 is little-endian, 1 is big-endian
    /// │ │ │ │ └──────── Reserved
    /// │ │ │ └────────── Reserved
    /// │ │ └──────────── Reserved
    /// │ └────────────── Reserved
    /// └──────────────── Reserved
    /// ```
    pub flag: u8,
}

impl ArchMode {

    pub fn new(flag: u8) -> Self {
        Self { flag }
    }

    pub fn width_flag(&self) -> u8 {
        self.flag & 0b0000_0011
    }

    pub fn width(&self) -> usize {
        match self.width_flag() {
            0b00 => 16,
            0b01 => 32,
            0b10 => 64,
            0b11 => 128,
            _ => unreachable!(),
        }
    }

    pub fn is_16bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b00
    }

    pub fn is_32bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b01
    }

    pub fn is_64bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b10
    }

    pub fn is_128bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b11
    }

    pub fn is_little_endian(&self) -> bool {
        (self.flag & 0b0000_0100) == 0
    }

    pub fn is_big_endian(&self) -> bool {
        (self.flag & 0b0000_0100) != 0
    }

    /// width to string
    pub fn width_to_string(&self) -> &str {
        match self.width_flag() {
            0b00 => "16",
            0b01 => "32",
            0b10 => "64",
            0b11 => "128",
            _ => "UNDEF",
        }
    }

    /// width from string
    pub fn width_from_string(s: &str) -> Self {
        let s = s.trim();
        match s {
            "16" => ArchMode::new(0b00),
            "32" => ArchMode::new(0b01),
            "64" => ArchMode::new(0b10),
            "128" => ArchMode::new(0b11),
            _ => ArchMode::new(0b01),
        }
    }

}

// ============================================================================== //
//                                info::ArchKind
// ============================================================================== //

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ArchKind {
    /// evo: ir arch
    EVO(ArchMode),
    /// risc-v
    RISCV(ArchMode),
    /// arm
    ARM(ArchMode),
    /// x86
    X86(ArchMode),
    /// mips
    MIPS(ArchMode),
    /// Undefined Arch Kind
    UNDEF,
}


impl ArchKind {


    pub fn mode(&self) -> ArchMode {
        match self {
            ArchKind::EVO(mode) => *mode,
            ArchKind::RISCV(mode) => *mode,
            ArchKind::ARM(mode) => *mode,
            ArchKind::X86(mode) => *mode,
            ArchKind::MIPS(mode) => *mode,
            _ => ArchMode::new(0b000),
        }
    }

    pub fn is_64bit(&self) -> bool {
        match self {
            ArchKind::EVO(mode) => mode.is_64bit(),
            ArchKind::RISCV(mode) => mode.is_64bit(),
            ArchKind::ARM(mode) => mode.is_64bit(),
            ArchKind::X86(mode) => mode.is_64bit(),
            ArchKind::MIPS(mode) => mode.is_64bit(),
            _ => false,
        }
    }

    pub fn is_32bit(&self) -> bool {
        match self {
            ArchKind::EVO(mode) => mode.is_32bit(),
            ArchKind::RISCV(mode) => mode.is_32bit(),
            ArchKind::ARM(mode) => mode.is_32bit(),
            ArchKind::X86(mode) => mode.is_32bit(),
            ArchKind::MIPS(mode) => mode.is_32bit(),
            _ => false,
        }
    }

    pub fn is_little_endian(&self) -> bool {
        match self {
            ArchKind::EVO(mode) => mode.is_little_endian(),
            ArchKind::RISCV(mode) => mode.is_little_endian(),
            ArchKind::ARM(mode) => mode.is_little_endian(),
            ArchKind::X86(mode) => mode.is_little_endian(),
            ArchKind::MIPS(mode) => mode.is_little_endian(),
            _ => false,
        }
    }

    pub fn is_big_endian(&self) -> bool {
        match self {
            ArchKind::EVO(mode) => mode.is_big_endian(),
            ArchKind::RISCV(mode) => mode.is_big_endian(),
            ArchKind::ARM(mode) => mode.is_big_endian(),
            ArchKind::X86(mode) => mode.is_big_endian(),
            ArchKind::MIPS(mode) => mode.is_big_endian(),
            _ => false,
        }
    }

    pub fn addr_scale(&self) -> usize {
        match self {
            ArchKind::EVO(mode) => mode.width(),
            ArchKind::RISCV(mode) => mode.width(),
            ArchKind::ARM(mode) => mode.width(),
            ArchKind::X86(mode) => mode.width(),
            ArchKind::MIPS(mode) => mode.width(),
            _ => 0,
        }
    }

    pub fn int_scale(&self) -> usize {
        match self {
            ArchKind::EVO(mode) => mode.width(),
            ArchKind::RISCV(mode) => mode.width(),
            ArchKind::ARM(mode) => mode.width(),
            ArchKind::X86(mode) => mode.width(),
            ArchKind::MIPS(mode) => mode.width(),
            _ => 0,
        }
    }

    pub fn float_scale(&self) -> usize {
        match self {
            ArchKind::EVO(mode) => mode.width(),
            ArchKind::RISCV(mode) => mode.width(),
            ArchKind::ARM(mode) => mode.width(),
            ArchKind::X86(mode) => mode.width(),
            ArchKind::MIPS(mode) => mode.width(),
            _ => 0,
        }
    }

    pub fn double_scale(&self) -> usize {
        match self {
            ArchKind::EVO(mode) => mode.width() * 2,
            ArchKind::RISCV(mode) => mode.width() * 2,
            ArchKind::ARM(mode) => mode.width() * 2,
            ArchKind::X86(mode) => mode.width() * 2,
            ArchKind::MIPS(mode) => mode.width() * 2,
            _ => 0,
        }
    }

    pub fn to_string (&self) -> String {
        match self {
            ArchKind::EVO(mode) => format!("evo{}", mode.width_to_string()),
            ArchKind::RISCV(mode) => format!("riscv{}", mode.width_to_string()),
            ArchKind::ARM(mode) => format!("arm{}", mode.width_to_string()),
            ArchKind::X86(mode) => format!("x86_{}", mode.width_to_string()),
            ArchKind::MIPS(mode) => format!("mips{}", mode.width_to_string()),
            _ => "UNDEF".to_string(),
        }
    }

    pub fn from_string(s: &str) -> ArchKind {
        let s = s.trim();
        if s.starts_with("evo") {
            ArchKind::EVO(ArchMode::width_from_string(&s[3..]))
        } else if s.starts_with("riscv") {
            ArchKind::RISCV(ArchMode::width_from_string(&s[5..]))
        } else if s.starts_with("arm") {
            ArchKind::ARM(ArchMode::width_from_string(&s[3..]))
        } else if s.starts_with("x86") {
            ArchKind::X86(ArchMode::width_from_string(&s[3..]))
        } else if s.starts_with("mips") {
            ArchKind::MIPS(ArchMode::width_from_string(&s[4..]))
        } else {
            ArchKind::UNDEF
        }
    }

}


impl fmt::Display for ArchKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}



// ============================================================================== //
//                                  info::Arch
// ============================================================================== //

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Arch {
    pub kind: ArchKind,
    pub reg_num: usize,
}


impl Arch {

    thread_local! {
        /// Arch Pool
        pub static ARCH_POOL: Rc<RefCell<HashMap<String, Arch>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    pub fn def(kind: ArchKind, reg_num: usize) -> Self {
        let name = ArchKind::to_string(&kind);
        let arch = Self {
            kind,
            reg_num,
        };
        Self::pool_nset(name.clone(), arch.clone());
        Self::pool_nget(name)
    }

    /// Def arch by string
    pub fn defs(s: &str, reg_num: usize) -> Self {
        Self::def(ArchKind::from_string(s), reg_num)
    }

    // ==================== arch.pool ====================== //

    /// Get Arch from pool by name
    pub fn pool_nget(name: String) -> Arch {
        Self::ARCH_POOL.with(|pool| {
            let pool = pool.borrow();
            pool.get(&name).map(|arch| arch.clone()).unwrap()
        })
    }

    /// Set Arch to pool by name
    pub fn pool_nset(name: String, arch: Arch) {
        Self::ARCH_POOL.with(|pool| {
            pool.borrow_mut().insert(name, arch);
        })
    }

    /// Delete Arch from pool by name
    pub fn pool_ndel(name: String) {
        Self::ARCH_POOL.with(|pool| {
            pool.borrow_mut().remove(&name);
        })
    }

    /// Clear Arch pool
    pub fn pool_clr() {
        Self::ARCH_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }

    /// Info of Arch pool
    pub fn pool_info() -> String {
        Self::ARCH_POOL.with(|pool| {
            let pool = pool.borrow();
            let mut info = String::new();
            info.push_str(format!("Arch Pool(Nums={}):\n", Self::pool_size()).as_str());
            for (_, arch) in pool.iter() {
                info += &format!("- {} ({}, {}, reg_num = {})\n", 
                    arch,
                    if arch.kind.is_64bit() { "64-bit" } else { "32-bit" },
                    if arch.kind.is_little_endian() { "Little-Endian" } else { "Big-Endian" },
                    arch.reg_num
                );
            }
            info
        })
    }

    pub fn pool_size() -> usize {
        Self::ARCH_POOL.with(|pool| {
            pool.borrow().len()
        })
    }


    pub fn to_string (&self) -> String {
        format!("{}", self.kind.to_string())
    }

    pub fn from_string(s: &str) -> Self {
        let s = s.trim();
        let kind = ArchKind::from_string(s);
        Arch::def(kind, 32)
    }
}


impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


// ============================================================================== //
//                                info::ArchInfo
// ============================================================================== //


/// `ArchInfo`: Config information of the architecture
pub trait ArchInfo {

    // ====================== Const ====================== //

    /// Arch name: like "evoir"
    const NAME: &'static str;
    /// Number of bytes in a byte: *1*, 2, 4
    const BYTE_SIZE: usize;
    /// Number of bytes in a addr(ptr/reg.size: 0x00 ~ 2^ADDR_SIZE): 8, 16, *32*, 64
    const ADDR_SIZE: usize;
    /// Number of bytes in a word(interger): 8, 16, **32**, 64
    const WORD_SIZE: usize;
    /// Number of bytes in a (float): **32**, 64
    const FLOAT_SIZE: usize;
    /// Base of Addr: 0x04000000
    const BASE_ADDR: usize;
    /// Mem size: default 4MB = 4 * 1024 * 1024
    const MEM_SIZE: usize;
    /// Stack Mem size: default 1MB = 1 * 1024 * 1024
    const STACK_SIZE: usize;
    /// Number of Registers: 8, 16, **32**, 64
    const REG_NUM: usize;

}





// ============================================================================== //
//                                Unit Tests
// ============================================================================== //


#[cfg(test)]
mod arch_info_test {

    use super::*;

    #[test]
    fn arch_kind() {

        let mode = ArchMode::new(ArchModeKind::L64 as u8);
        assert_eq!(mode.is_64bit(), true);
        assert_eq!(mode.is_32bit(), false);
        assert_eq!(mode.is_128bit(), false);
        assert_eq!(mode.is_little_endian(), true);
        assert_eq!(mode.is_big_endian(), false);
        let kind = ArchKind::EVO(mode);
        assert_eq!(kind.to_string(), "evo64");

        let kind2 = ArchKind::RISCV(ArchMode::new(0b001));
        assert_eq!(kind2.to_string(), "riscv32");

        let kind3 = ArchKind::X86(ArchMode::new(0b010));
        assert_eq!(kind3.to_string(), "x86_64");

        let kind4 = ArchKind::from_string("evo64");
        assert_eq!(kind4, ArchKind::EVO(ArchMode::new(ArchModeKind::L64 as u8)));
    }


    #[test]
    fn arch_from() {

        let arch = Arch::defs("evo64", 32);
        println!("{}", arch);
        println!("{}", Arch::pool_info());
    }
}