//! `evo::ir::itp`: IR Interpreter
//! 
//!


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


use crate::ir::op::IRInsn;
use crate::ir::ctx::IRContext;


/// IR Interpreter
pub struct IRInterpreter {
    /// Context of IR
    pub ctx : Rc<RefCell<IRContext>>,

    /// Interface Function Table
    pub ift : Rc<RefCell<HashMap<usize, String>>>,

    /// Branch Instructions (For Branch Optimization)
    pub bis : Rc<RefCell<Vec<IRInsn>>>,

    /// Performence Tools
    #[cfg(feature = "perf")]
    pub insn_ident_size : usize
}


impl IRInterpreter {

    /// Init a IRInterpreter
    pub fn init(ctx : Rc<RefCell<IRContext>>) -> IRInterpreter {
        let v = Self { 
            ctx,
            ift : Rc::new(RefCell::new(HashMap::new())),
            bis : Rc::new(RefCell::new(Vec::new())),

            #[cfg(feature = "perf")]
            insn_ident_size : 0
        };

        v
    }


}

