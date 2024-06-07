//! `evo::ir::syc` : Syscall (Unsafe)
//! 
//!




// ============================================================================== //
//                                 Use Mods
// ============================================================================== //
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use libc::c_void;

use crate::core::cpu::CPUState;

// ============================================================================== //
//                             syc::Syscaller
// ============================================================================== //


/// Syscall: system call
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Syscaller {

}



impl Syscaller {

    thread_local! {
        /// Syscall Map
        pub static SYSCALL_MAP: Rc<RefCell<HashMap<usize, &'static str>>> = Rc::new(RefCell::new(HashMap::new()));
        /// Syscall Pool
        pub static SYSCALL_POOL: Rc<RefCell<HashMap<&'static str, (&'static str, fn(&CPUState) -> u64)>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    /// Define syscall
    pub fn def(name: &'static str, call: fn(&CPUState) -> u64) {
        Self::pool_nset(name, call);
    }

    /// Call syscall
    pub fn call(cpu: &CPUState, name: &'static str) -> u64 {
        Self::pool_nget(name).1(cpu)
    }

    // =================== Syscall.ctl ======================= //

    /// Init a Syscaller pool
    pub fn pool_init() {
        Syscaller::def("exit",
            |cpu| {
                // ========= Exit Program ========= //
                // 1. Get x10/a0 Reg
                let code = cpu.get_nreg("x10").get_i32(0);
                // 2. Exit
                unsafe { libc::exit(code) };
            }
        );
        Syscaller::def("close",
            |cpu| {
                // ========= Close File ========= //
                // 1. Get x10/a0 Reg
                let fd = cpu.get_nreg("x10").get_i32(0);
                // 2. Close file
                let ret;
                unsafe { ret = libc::close(fd) };
                // 3. Return
                ret as u64
            }
        );
        Syscaller::def("read",
            |cpu| {
                // ========= Read File ========= //
                // 1. Get x10/a0 Reg
                let fd = cpu.get_nreg("x10").get_i32(0);
                // 2. Get x11/a1 Reg
                let bufptr = cpu.get_nreg("x11").get_i32(0) as *mut c_void;
                // 3. Get x12/a2 Reg
                let count = cpu.get_nreg("x12").get_i32(0) as usize;
                // 4. Read file
                let ret;
                unsafe { ret = libc::read(fd, bufptr, count) };
                // 5. Return
                ret as u64
            }
        );
        Syscaller::def("write",
            |cpu| {
                // ========= Write File ========= //
                // 1. Get x10/a0 Reg
                let fd = cpu.get_nreg("x10").get_i32(0);
                // 2. Get x11/a1 Reg
                let bufptr = cpu.get_nreg("x11").get_i32(0) as *const c_void;
                // 3. Get x12/a2 Reg
                let count = cpu.get_nreg("x12").get_i32(0) as usize;
                // 4. Write file
                let ret;
                unsafe { ret = libc::write(fd, bufptr, count) };
                // 5. Return
                ret as u64
            }
        );
        Syscaller::def("fstat",
            |cpu| {
                // ========= Stat File ========= //
                // 1. Get x10/a0 Reg
                let fd = cpu.get_nreg("x10").get_i32(0);
                // 2. Get x11/a1 Reg
                let buf = cpu.get_nreg("x11").get_i32(0) as *mut libc::stat;
                // 3. Stat file
                let ret;
                unsafe { ret = libc::fstat(fd, buf) };
                // 4. Return 0
                ret as u64
            }
        );
        Syscaller::def("gettimeofday",
            |cpu| {
                // ========= Get Time ========= //
                // 1. Get x10/a0 Reg
                let tv = cpu.get_nreg("x10").get_i32(0) as *mut libc::timeval;
                // 2. Get x11/a1 Reg
                let tz = cpu.get_nreg("x11").get_i32(0) as *mut libc::timezone;
                // 3. Get Time
                let ret;
                unsafe { ret = libc::gettimeofday(tv, tz) };
                // 4. Return 0
                ret as u64
            }
        );
        Syscaller::def("brk",
            |cpu| {
                // ========= Get Time ========= //
                // 1. Get x10/a0 Reg
                let addr = cpu.get_nreg("x10").get_i32(0) as *mut c_void;
                // 2. Get x11/a1 Reg
                let ret;
                unsafe { ret = libc::brk(addr) };
                // 3. Return 0
                ret as u64
            }
        )
    }

    /// Get syscall
    pub fn pool_nget(name: &'static str) -> (&'static str, fn(&CPUState) -> u64) {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow().get(name).unwrap().clone()
        })
    }

    /// Set syscall
    pub fn pool_nset(name: &'static str, call: fn(&CPUState) -> u64) {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow_mut().insert(name, (name, call));
        })
    }

    /// Delete syscall
    pub fn pool_ndel(name: &'static str) {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow_mut().remove(name);
        })
    }

    /// Clear syscall
    pub fn pool_clr() {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow_mut().clear();
        })
    }

    /// Get syscall pool size
    pub fn pool_size() -> usize {
        Self::SYSCALL_POOL.with(|pool| pool.borrow().len())
    }

    /// Check is in pool
    pub fn pool_is_in(name: &'static str) -> bool {
        Self::SYSCALL_POOL.with(|pool| pool.borrow().get(name).is_some())
    }

    /// Pool Info
    pub fn pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Syscalls (Num = {}):\n", Self::pool_size()));
        Self::SYSCALL_POOL.with(|pool| {
            let mut i = 0;
            for (name, _) in pool.borrow().iter() {
                info.push_str(&format!("[{:>4}] {}\n", i, name));
                i += 1;
            }
        });
        info
    }


}




#[cfg(test)]
mod syscall_test {

    use super::*;

    #[test]
    fn syc_info() {
        Syscaller::pool_init();
        println!("{}", Syscaller::pool_info());
    }
}