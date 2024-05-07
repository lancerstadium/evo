//! `evo::ir::itp`: IR Interpreter
//! 
//!


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::arch::info::Arch;
use crate::ir::cpu::CPUState;
use crate::ir::insn::Instruction;



// ============================================================================== //
//                               itp::Interpreter
// ============================================================================== //

/// IR Interpreter
#[derive(Clone, PartialEq)]
pub struct Interpreter {
    /// arch
    pub arch: Arch,
    /// Interface Function Table
    pub ift : Rc<RefCell<HashMap<usize, String>>>,
    /// Branch Instructions (For Branch Optimization)
    pub bis : Rc<RefCell<Vec<Instruction>>>,
    /// Performence Tools
    #[cfg(feature = "perf")]
    pub insn_num : usize
}


impl Interpreter {

    thread_local! {
        /// HashMap of insn interpreter functions pool
        pub static INTERP_FUNC_POOL: Rc<RefCell<HashMap<&'static str, fn(&CPUState, &Instruction)>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    // =================== IRItp.ctl ======================= //

    /// Init a Interpreter
    pub fn init(arch: Arch) -> Interpreter {
        let v = Self { 
            arch,
            ift : Rc::new(RefCell::new(HashMap::new())),
            bis : Rc::new(RefCell::new(Vec::new())),

            #[cfg(feature = "perf")]
            insn_ident_size : 0
        };
        v
    }

    /// Define insn with func
    pub fn def_insn(&self, name: &'static str, syms: Vec<i32>, ty: &'static str, opb: &'static str, func: fn(&CPUState, &Instruction)) {
        Instruction::def(&self.arch, name, syms, ty, opb);
        Self::def(name, func);
    }

    /// Define an IRItp func
    pub fn def(name: &'static str, func: fn(&CPUState, &Instruction)) {
        Self::pool_set(name, func);
    }

    /// Execute an instruction
    pub fn execute(&self, cpu: &CPUState, insn: &Instruction) {
        // 1. Get IRItp func
        let func = Self::pool_get(insn.name());
        // 2. Execute
        func(cpu, insn);
    }


    // =================== IRItp.pool ====================== //

    /// Get IRItp by name
    pub fn pool_get(name: &'static str) -> fn(&CPUState, &Instruction) {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow().get(name).unwrap().clone()
        })
    }

    /// Set IRItp by name
    pub fn pool_set(name: &'static str, func: fn(&CPUState, &Instruction)) {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow_mut().insert(name, func);
        })
    }

    /// Delete IRItp by name
    pub fn pool_del(name: &'static str) {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow_mut().remove(name);
        })
    }

    /// Clear IRItp pool
    pub fn pool_clr() {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }

    /// Get IRItp pool size
    pub fn pool_size() -> usize {
        Self::INTERP_FUNC_POOL.with(|pool| pool.borrow().len())
    }

    /// Check is in pool
    pub fn pool_is_in(name: &'static str) -> bool {
        Self::INTERP_FUNC_POOL.with(|pool| pool.borrow().get(name).is_some())
    }

    /// Info of IRItp pool
    pub fn pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Interpreter pool info: \n"));
        info.push_str(&format!("  - size: {}\n", Self::pool_size()));
        info
    }



}

