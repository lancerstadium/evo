

// ============================================================================== //
//                                info::ArchInfo
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