




// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


use crate::arch::riscv::{def::RISCV32_ARCH, trs::riscv32_trs_init};
use crate::log_warning;
use crate::util::log::Span;
use crate::arch::info::Arch;
use crate::core::insn::Instruction;

use super::insn::RegFile;
use super::op::Operand;


// ============================================================================== //
//                                trs::Translator
// ============================================================================== //


/// Arch Interpreter
#[derive(Debug, Clone, PartialEq)]
pub struct Translator {
    /// src arch
    pub src_arch: &'static Arch,
    /// trg arch
    pub trg_arch: &'static Arch,

}


impl Translator {

    thread_local! {
        /// HashMap of insn translator functions pool
        pub static TRANS_FUNC_POOL: Rc<RefCell<HashMap<(&'static Arch, &'static Arch, &'static str), fn(&Instruction) -> Vec<Instruction>>>> = Rc::new(RefCell::new(HashMap::new()));
        pub static TRANS_POOL: Rc<RefCell<HashMap<&'static Arch, Rc<RefCell<Translator>>>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    pub fn def(src_arch: &'static Arch, trg_arch: &'static Arch) -> Rc<RefCell<Translator>> {
        let v = Self { src_arch, trg_arch };
        Self::trs_pool_set(src_arch, Rc::new(RefCell::new(v)));
        Self::trs_pool_get(src_arch)
    }

    pub fn def_func(&self, insn_name: &'static str, func: fn(&Instruction) -> Vec<Instruction>) {
        Self::func_pool_nset(self.src_arch, self.trg_arch, insn_name, func);
    }

    /// Translate insn
    pub fn translate(&self, insn: &Instruction) -> Vec<Instruction> {
        Self::func_pool_nget(self.src_arch, self.trg_arch, insn.name())(insn)
    }

    // =================== Trans.pool ====================== //

    /// route of trs pool
    pub fn trs_pool_init(src_arch: &'static Arch, trg_arch: &'static Arch) -> Option<Rc<RefCell<Translator>>> {
        match *src_arch {
            RISCV32_ARCH => riscv32_trs_init(trg_arch),
            _ => {
                log_warning!("Translator init fail, not support arch: {}", src_arch.name);
                None
            }
        }
    }

    pub fn trs_pool_is_in(arch: &'static Arch) -> bool {
        Self::TRANS_POOL.with(|pool| {
            pool.borrow().get(arch).is_some()
        })
    }

    pub fn trs_pool_size() -> usize {
        Self::TRANS_POOL.with(|pool| {
            pool.borrow().len()
        })
    }

    pub fn trs_pool_set(arch: &'static Arch, trs: Rc<RefCell<Translator>>) {
        Self::TRANS_POOL.with(|pool| {
            pool.borrow_mut().insert(arch, trs);
        })
    }

    pub fn trs_pool_get(arch: &'static Arch) -> Rc<RefCell<Translator>> {
        Self::TRANS_POOL.with(|pool| {
            pool.borrow().get(arch).unwrap().clone()
        })
    }

    pub fn trs_pool_del(arch: &'static Arch) {
        Self::TRANS_POOL.with(|pool| {
            pool.borrow_mut().remove(arch);
        })
    }

    pub fn trs_pool_clr() {
        Self::TRANS_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }

    // ==================== Func.pool ====================== //

    pub fn func_pool_is_in(src_arch: &'static Arch, trg_arch: &'static Arch, insn_name: &'static str) -> bool {
        Self::TRANS_FUNC_POOL.with(|pool| {
            pool.borrow().get(&(src_arch, trg_arch, insn_name)).is_some()
        })
    }

    pub fn func_pool_size() -> usize {
        Self::TRANS_FUNC_POOL.with(|pool| {
            pool.borrow().len()
        })
    }

    pub fn func_pool_nset(src_arch: &'static Arch, trg_arch: &'static Arch, insn_name: &'static str, func: fn(&Instruction) -> Vec<Instruction>) {
        Self::TRANS_FUNC_POOL.with(|pool| {
            pool.borrow_mut().insert((src_arch, trg_arch, insn_name), func);
        })
    }

    pub fn func_pool_nget(src_arch: &'static Arch, trg_arch: &'static Arch, insn_name: &'static str) -> fn(&Instruction) -> Vec<Instruction> {
        Self::TRANS_FUNC_POOL.with(|pool| {
            pool.borrow().get(&(src_arch, trg_arch, insn_name)).unwrap().clone()
        })
    }

    pub fn func_pool_ndel(src_arch: &'static Arch, trg_arch: &'static Arch, insn_name: &'static str) {
        Self::TRANS_FUNC_POOL.with(|pool| {
            pool.borrow_mut().remove(&(src_arch, trg_arch, insn_name));
        })
    }

    pub fn func_pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Translator Func Pool(Nums={}): \n", Self::func_pool_size()));
        // iter hashmap
        Self::TRANS_FUNC_POOL.with(|pool| {
            for (k, _) in pool.borrow().iter() {
                info.push_str(&format!("- {:<10} [{} -> {}]\n", k.2, k.0, k.1));
            }
        });
        info
    }

    pub fn func_pool_clr() {
        Self::TRANS_FUNC_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }
}