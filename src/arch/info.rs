
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::{default, fmt};
use std::rc::Rc;
use std::cell::RefCell;

use super::evo::def::EVO_ARCH;


// ============================================================================== //
//                                info::ArchMode
// ============================================================================== //

#[derive(Hash, Debug, PartialEq, Eq, Clone, Copy)]
pub struct ArchMode {
    /// ### Arch Mode flag
    /// ```txt
    /// (0-7):
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ │ ├─┘
    /// │ │ │ │ │ │ └──── (0-1) 00 is 16-bit, 01 is 32-bit, 10 is 64-bit, 11 is 128-bit
    /// │ │ │ │ │ └────── (2) 0 is little-endian, 1 is big-endian
    /// │ │ │ │ └──────── <Reserved>
    /// │ │ │ └────────── <Reserved>
    /// │ │ └──────────── <Reserved>
    /// │ └────────────── <Reserved>
    /// └──────────────── <Reserved>
    /// ```
    pub flag: u16,

}
/// Arch mode width 128
pub const BIT128: u16 = 0b11;
/// Arch mode width 64
pub const BIT64: u16 = 0b10;
/// Arch mode width 32
pub const BIT32: u16 = 0b01;
/// Arch mode width 16
pub const BIT16: u16 = 0b00;
/// Arch mode endianness little
pub const LITTLE_ENDIAN: u16 = 0b000;
/// Arch mode endianness big
pub const BIG_ENDIAN: u16 = 0b100;

impl ArchMode {
    
    pub const fn new(flag: u16) -> Self {
        Self { flag }
    }

    pub const fn width_flag(&self) -> u16 {
        self.flag & 0b0000_0011
    }

    pub const fn width(&self) -> usize {
        match self.width_flag() {
            0b00 => 16,
            0b01 => 32,
            0b10 => 64,
            0b11 => 128,
            _ => unreachable!(),
        }
    }

    pub const fn is_16bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b00
    }

    pub const fn is_32bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b01
    }

    pub const fn is_64bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b10
    }

    pub const fn is_128bit(&self) -> bool {
        (self.flag & 0b0000_0011) == 0b11
    }

    pub const fn is_little_endian(&self) -> bool {
        (self.flag & 0b0000_0100) == 0
    }

    pub const fn is_big_endian(&self) -> bool {
        (self.flag & 0b0000_0100) != 0
    }

    pub const fn endian_to_string(&self) -> &str {
        match self.is_little_endian() {
            true => "little-endian",
            false => "big-endian",
        }
    }

    /// width to string
    pub const fn width_to_string(&self) -> &str {
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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum ArchKind {
    /// evo: ir arch
    EVO,
    /// risc-v
    RISCV,
    /// arm
    ARM,
    /// x86
    X86,
    /// mips
    MIPS,
    /// Undefined Arch Kind
    UNDEF,
}


impl ArchKind {

    pub const fn to_string(&self) -> &'static str {
        match self {
            ArchKind::EVO => "evo",
            ArchKind::RISCV => "riscv",
            ArchKind::ARM => "arm",
            ArchKind::X86 => "x86",
            ArchKind::MIPS => "mips",
            _ => "UNDEF",
        }
    }

    pub fn from_string(s: &str) -> ArchKind {
        let s = s.trim();
        if s.starts_with("evo") {
            ArchKind::EVO
        } else if s.starts_with("riscv") {
            ArchKind::RISCV
        } else if s.starts_with("arm") {
            ArchKind::ARM
        } else if s.starts_with("x86") {
            ArchKind::X86
        } else if s.starts_with("mips") {
            ArchKind::MIPS
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

#[derive(Hash, Debug, PartialEq, Eq, Clone)]
pub struct Arch {
    pub name: &'static str,
    pub kind: ArchKind,
    pub mode: ArchMode,
    pub reg_num: usize,
}


impl Arch {

    thread_local! {
        pub static ARCH_POOL: Rc<RefCell<Vec<Arch>>> = Rc::new(RefCell::new(Vec::new()));
    }


    // ==================== Arch.ctl ======================= //

    pub const fn new(kind: ArchKind, mode_flag: u16, reg_num: usize) -> Self {
        let mode = ArchMode::new(mode_flag);
        let name = ArchKind::to_string(&kind);
        Self {
            name,
            kind,
            mode,
            reg_num,
        }
    }

    pub fn def(arch: Arch) -> Self {
        Self::pool_push(arch);
        Self::pool_last()
    }

    pub const fn addr_scale(&self) -> usize {
        self.mode.width()
    }

    pub const fn int_scale(&self) -> usize {
        self.mode.width()
    }

    pub const fn float_scale(&self) -> usize {
        self.mode.width()
    }

    pub const fn double_scale(&self) -> usize {
        self.mode.width() * 2
    }


    // =================== Arch.pool ======================= //

    pub fn pool_push(arch: Arch) -> usize {
        Self::ARCH_POOL.with(|pool| pool.borrow_mut().push(arch));
        Self::ARCH_POOL.with(|pool| pool.borrow().len() - 1)
    }

    pub fn pool_get(idx: usize) -> Arch {
        Self::ARCH_POOL.with(|pool| pool.borrow().get(idx).unwrap().clone())
    }

    pub fn pool_set(idx: usize, arch: Arch) {
        Self::ARCH_POOL.with(|pool| pool.borrow_mut().get_mut(idx).unwrap().clone_from(&arch));
    }

    pub fn pool_last() -> Arch {
        Self::ARCH_POOL.with(|pool| pool.borrow().last().unwrap().clone())
    }

    pub fn pool_del(idx: usize) {
        Self::ARCH_POOL.with(|pool| pool.borrow_mut().remove(idx));
    }

    pub fn pool_clr() {
        Self::ARCH_POOL.with(|pool| pool.borrow_mut().clear());
    }

    pub fn pool_size() -> usize {
        Self::ARCH_POOL.with(|pool| pool.borrow().len())
    }

    pub fn pool_info() -> String {
        let mut str = String::new();
        str.push_str(format!("Arch Pool(Nums={}):\n", Self::pool_size()).as_str());
        for i in 0..Self::pool_size() {
            let arch = Self::pool_get(i);
            str.push_str(format!("{}: {} (width={}, {}, reg_num={})\n", 
                i, 
                arch,
                arch.mode.width_to_string(),
                arch.mode.endian_to_string(),
                arch.reg_num
            ).as_str());
        }
        str
    }


    pub fn to_string (&self) -> String {
        match self.kind {
            ArchKind::EVO => format!("evo"),
            ArchKind::RISCV => format!("riscv{}", self.mode.width_to_string()),
            ArchKind::ARM => format!("arm{}", self.mode.width_to_string()),
            ArchKind::X86 => format!("x86_{}", self.mode.width_to_string()),
            ArchKind::MIPS => format!("mips{}", self.mode.width_to_string()),
            _ => "UNDEF".to_string(),
        }
    }

}


impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl default::Default for Arch {
    fn default() -> Self {
        EVO_ARCH
    }
}






// ============================================================================== //
//                                Unit Tests
// ============================================================================== //


#[cfg(test)]
mod arch_info_test {

    use super::*;

    #[test]
    fn arch_kind() {

        let mode = ArchMode::new(0b010);
        assert_eq!(mode.is_64bit(), true);
        assert_eq!(mode.is_32bit(), false);
        assert_eq!(mode.is_128bit(), false);
        assert_eq!(mode.is_little_endian(), true);
        assert_eq!(mode.is_big_endian(), false);
    }


    #[test]
    fn arch_from() {

        let arch = Arch::new(ArchKind::RISCV, BIT32 | LITTLE_ENDIAN, 32);
        let arch = Arch::def(arch);
        println!("{}", arch);
        println!("{}", Arch::pool_info());
    }
}