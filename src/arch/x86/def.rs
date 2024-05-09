

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::{Arch, ArchKind, BIT32, BIT64, LITTLE_ENDIAN};
use crate::ir::val::Value;
use crate::ir::insn::{Instruction, OPRS_SIG, OPRS_USD};
use crate::ir::itp::Interpreter;
use crate::ir::mem::CPUThreadStatus;




// ============================================================================== //
//                             x86::def::arch
// ============================================================================== //


/// `i386`
/// ### Registers
/// 
/// ```txt
/// ┌────────┬────────┬────────┬────────┬───────┐
/// │ Encode │ [31:0] │ [15:0] │ [15:8] │ [7:0] │
/// ├────────┼────────┼────────┼────────┼───────┤
/// │  000   │  eax   │   ax   │   ah   │  al   │  
/// │  001   │  ecx   │   cx   │   ch   │  cl   │
/// │  010   │  edx   │   dx   │   dh   │  dl   │
/// │  011   │  ebx   │   bx   │   bh   │  bl   │
/// │  100   │  esp   │   sp   │   --   │  --   │
/// │  101   │  ebp   │   bp   │   --   │  --   │
/// │  110   │  esi   │   si   │   --   │  --   │
/// │  111   │  edi   │   di   │   --   │  --   │
/// └────────┴────────┴────────┴────────┴───────┘
/// ```
pub const X86_ARCH: Arch = Arch::new(ArchKind::X86, BIT32 | LITTLE_ENDIAN, 8);



// ============================================================================== //
//                          evo::def::interpreter
// ============================================================================== //

/// Insn temp and Reg and Interpreter Pool Init
pub fn x86_itp_init() -> Option<Rc<RefCell<Interpreter>>> {
    // 1. Init regs pool
    Instruction::reg("eax", Value::bit(3, 0));
    Instruction::reg("ecx", Value::bit(3, 1));
    Instruction::reg("edx", Value::bit(3, 2));
    Instruction::reg("ebx", Value::bit(3, 3));
    Instruction::reg("esp", Value::bit(3, 4));
    Instruction::reg("ebp", Value::bit(3, 5));
    Instruction::reg("esi", Value::bit(3, 6));
    Instruction::reg("edi", Value::bit(3, 7));

    Instruction::reg("ax", Value::bit(3, 0));
    Instruction::reg("cx", Value::bit(3, 1));
    Instruction::reg("dx", Value::bit(3, 2));
    Instruction::reg("bx", Value::bit(3, 3));
    Instruction::reg("sp", Value::bit(3, 4));
    Instruction::reg("bp", Value::bit(3, 5));
    Instruction::reg("si", Value::bit(3, 6));
    Instruction::reg("di", Value::bit(3, 7));

    Instruction::reg("ah", Value::bit(3, 0));
    Instruction::reg("ch", Value::bit(3, 1));
    Instruction::reg("dh", Value::bit(3, 2));
    Instruction::reg("bh", Value::bit(3, 3));

    Instruction::reg("al", Value::bit(3, 0));
    Instruction::reg("cl", Value::bit(3, 1));
    Instruction::reg("dl", Value::bit(3, 2));
    Instruction::reg("bl", Value::bit(3, 3));

    // 2. Init insns & insns interpreter
    let itp = Interpreter::def(&X86_ARCH);

    itp.borrow_mut().def_insn("mov", BIT32 | LITTLE_ENDIAN, vec![1, 1], "X", "0x88", 
        |cpu, insn| {
            
        }
    );

    Some(itp)
}


#[cfg(test)]
mod x86_test {

    use super::*;
    use crate::ir::cpu::CPUState;


    #[test]
    fn x86_itp() {
        let cpu = CPUState::init(&X86_ARCH, &X86_ARCH, None, None, None);
        println!("{}", CPUState::pool_info());
    }

}