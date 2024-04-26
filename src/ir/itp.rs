//! `evo::ir::itp`: IR Interpreter
//! 
//!


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::cell::RefCell;


use crate::ir::ctx::IRContext;


/// IR Interpreter
pub struct IRInterpreter {
    /// Context of IR
    pub ctx : RefCell<IRContext>,
}


impl IRInterpreter {

    /// Create a IRInterpreter
    pub fn new(ctx : RefCell<IRContext>) -> IRInterpreter {
        Self { ctx }
    }
}

