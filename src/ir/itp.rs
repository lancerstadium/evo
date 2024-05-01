//! `evo::ir::itp`: IR Interpreter
//! 
//!


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::ir::ctx::IRContext;
use crate::ir::op::IRInsn;



// ============================================================================== //
//                               itp::IRInterpreter
// ============================================================================== //

/// IR Interpreter
#[derive(Clone, PartialEq)]
pub struct IRInterpreter {
    /// Interface Function Table
    pub ift : Rc<RefCell<HashMap<usize, String>>>,
    /// Branch Instructions (For Branch Optimization)
    pub bis : Rc<RefCell<Vec<IRInsn>>>,
    /// Performence Tools
    #[cfg(feature = "perf")]
    pub insn_num : usize
}


impl IRInterpreter {

    thread_local! {
        /// HashMap of insn interpreter functions pool
        pub static IR_INTERP_POOL: Rc<RefCell<HashMap<&'static str, fn(&IRContext, &IRInsn)>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    // =================== IRItp.ctl ======================= //

    /// Init a IRInterpreter
    pub fn init() -> IRInterpreter {
        let v = Self { 
            ift : Rc::new(RefCell::new(HashMap::new())),
            bis : Rc::new(RefCell::new(Vec::new())),

            #[cfg(feature = "perf")]
            insn_ident_size : 0
        };
        v
    }

    /// Define insn with func
    pub fn def_insn(name: &'static str, syms: Vec<i32>, ty: &'static str, opb: &'static str, func: fn(&IRContext, &IRInsn)) {
        IRInsn::def(name, syms, ty, opb);
        Self::def(name, func);
    }

    /// Define an IRItp func
    pub fn def(name: &'static str, func: fn(&IRContext, &IRInsn)) {
        Self::pool_set(name, func);
    }

    /// Execute an instruction
    pub fn execute(&self, ctx: &IRContext, insn: &IRInsn) {
        // 1. Get IRItp func
        let func = Self::pool_get(insn.name());
        // 2. Execute
        func(ctx, insn);
    }


    // =================== IRItp.pool ====================== //

    /// Get IRItp by name
    pub fn pool_get(name: &'static str) -> fn(&IRContext, &IRInsn) {
        Self::IR_INTERP_POOL.with(|pool| {
            pool.borrow().get(name).unwrap().clone()
        })
    }

    /// Set IRItp by name
    pub fn pool_set(name: &'static str, func: fn(&IRContext, &IRInsn)) {
        Self::IR_INTERP_POOL.with(|pool| {
            pool.borrow_mut().insert(name, func);
        })
    }

    /// Delete IRItp by name
    pub fn pool_del(name: &'static str) {
        Self::IR_INTERP_POOL.with(|pool| {
            pool.borrow_mut().remove(name);
        })
    }

    /// Clear IRItp pool
    pub fn pool_clr() {
        Self::IR_INTERP_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }

    /// Get IRItp pool size
    pub fn pool_size() -> usize {
        Self::IR_INTERP_POOL.with(|pool| pool.borrow().len())
    }

    /// Check is in pool
    pub fn pool_is_in(name: &'static str) -> bool {
        Self::IR_INTERP_POOL.with(|pool| pool.borrow().get(name).is_some())
    }

    /// Info of IRItp pool
    pub fn pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("IRInterpreter pool info: \n"));
        info.push_str(&format!("  - size: {}\n", Self::pool_size()));
        info
    }



}

