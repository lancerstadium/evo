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
    pub arch: &'static Arch,
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
        pub static INTERP_FUNC_POOL: Rc<RefCell<HashMap<(&'static Arch, &'static str), fn(&CPUState, &Instruction)>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    // =================== IRItp.ctl ======================= //

    /// Init a Interpreter
    pub fn init(arch: &'static Arch) -> Interpreter {
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
    pub fn def_insn(&self, insn_name: &'static str, syms: Vec<i32>, ty: &'static str, opb: &'static str, func: fn(&CPUState, &Instruction)) {
        Instruction::def(self.arch, insn_name, syms, ty, opb);
        self.def(insn_name, func);
    }

    /// Define an IRItp func
    pub fn def(&self, insn_name: &'static str, func: fn(&CPUState, &Instruction)) {
        Self::func_pool_nset(self.arch, insn_name, func);
    }

    /// Execute an instruction
    pub fn execute(&self, cpu: &CPUState, insn: &Instruction) {
        // 1. Get IRItp func
        let func = Self::func_pool_nget(self.arch, insn.name());
        // 2. Execute
        func(cpu, insn);
    }


    // =================== IRItp.pool ====================== //

    /// Get IRItp by name
    pub fn func_pool_nget(arch: &'static Arch, insn_name: &'static str) -> fn(&CPUState, &Instruction) {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow().get(&(arch, insn_name)).unwrap().clone()
        })
    }

    /// Set IRItp by name
    pub fn func_pool_nset(arch: &'static Arch, insn_name: &'static str, func: fn(&CPUState, &Instruction)) {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow_mut().insert((arch, insn_name), func);
        })
    }

    /// Delete IRItp by name
    pub fn func_pool_ndel(arch: &'static Arch, insn_name: &'static str) {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow_mut().remove(&(arch, insn_name));
        })
    }

    /// Check is in pool
    pub fn func_pool_is_in(arch: &'static Arch, insn_name: &'static str) -> bool {
        Self::INTERP_FUNC_POOL.with(|pool| pool.borrow().get(&(arch, insn_name)).is_some())
    }


    /// Get IRItp pool size
    pub fn func_pool_size() -> usize {
        Self::INTERP_FUNC_POOL.with(|pool| pool.borrow().len())
    }

    /// Info of IRItp pool
    pub fn func_pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Interpreter Func Pool(Nums={}): \n", Self::func_pool_size()));
        // iter hashmap
        Self::INTERP_FUNC_POOL.with(|pool| {
            for (k, _) in pool.borrow().iter() {
                info.push_str(&format!("- {:<10} [{}]\n", k.1, k.0));
            }
        });
        info
    }

    /// Clear IRItp pool
    pub fn func_pool_clr() {
        Self::INTERP_FUNC_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }
    


}

