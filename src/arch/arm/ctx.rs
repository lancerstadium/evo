


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;

use crate::ir::mem::IRProcess;
use crate::ir::itp::IRInterpreter;

// ============================================================================== //
//                             arm::Arm32Context
// ============================================================================== //

/// `Arm32Context`: Context of ARM32 architecture
#[derive(Clone, PartialEq)]
pub struct Arm32Context {
    /// `proc`: Process Handle
    pub proc: Rc<RefCell<IRProcess>>,
    /// `itp`: Interpreter
    pub itp: Option<Rc<RefCell<IRInterpreter>>>,
}