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

use crate::ir::ctx::IRContext;

// ============================================================================== //
//                             syc::Syscaller
// ============================================================================== //


/// Syscall: system call
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Syscaller {

}



impl Syscaller {

    thread_local! {
        /// Syscall Pool
        pub static SYSCALL_POOL: Rc<RefCell<HashMap<usize, (&'static str, fn(&IRContext) -> u64)>>> = Rc::new(RefCell::new(HashMap::new()));
    }

    /// Define syscall
    pub fn def(id : usize, name: &'static str, call: fn(&IRContext) -> u64) {
        Self::pool_set(id, name, call)
    }

    /// Call syscall
    pub fn call(ctx: &IRContext, syscall_id: usize) -> u64 {
        Self::pool_get(syscall_id).1(ctx)
    }

    // =================== Syscall.ctl ======================= //

    /// Init a Syscaller pool
    pub fn pool_init() {
        Syscaller::def(93, "exit",
            |ctx| {
                // ========= Exit Program ========= //
                // 1. Get x10/a0 Reg
                let code = ctx.get_nreg("x10").get_i32(0);
                // 2. Exit
                unsafe { libc::exit(code) };
            }
        );
        Syscaller::def(57, "close",
            |ctx| {
                // ========= Close File ========= //
                // 1. Get x10/a0 Reg
                let fd = ctx.get_nreg("x10").get_i32(0);
                // 2. Close file
                let ret;
                unsafe { ret = libc::close(fd) };
                // 3. Return
                ret as u64
            }
        );
        Syscaller::def(63, "read",
            |ctx| {
                // ========= Read File ========= //
                // 1. Get x10/a0 Reg
                let fd = ctx.get_nreg("x10").get_i32(0);
                // 2. Get x11/a1 Reg
                let bufptr = ctx.get_nreg("x11").get_i32(0) as *mut c_void;
                // 3. Get x12/a2 Reg
                let count = ctx.get_nreg("x12").get_i32(0) as usize;
                // 4. Read file
                let ret;
                unsafe { ret = libc::read(fd, bufptr, count) };
                // 5. Return
                ret as u64
            }
        );
        Syscaller::def(64, "write",
            |ctx| {
                // ========= Write File ========= //
                // 1. Get x10/a0 Reg
                let fd = ctx.get_nreg("x10").get_i32(0);
                // 2. Get x11/a1 Reg
                let bufptr = ctx.get_nreg("x11").get_i32(0) as *const c_void;
                // 3. Get x12/a2 Reg
                let count = ctx.get_nreg("x12").get_i32(0) as usize;
                // 4. Write file
                let ret;
                unsafe { ret = libc::write(fd, bufptr, count) };
                // 5. Return
                ret as u64
            }
        );
        Syscaller::def(80, "fstat",
            |ctx| {
                // ========= Stat File ========= //
                // 1. Get x10/a0 Reg
                let fd = ctx.get_nreg("x10").get_i32(0);
                // 2. Get x11/a1 Reg
                let buf = ctx.get_nreg("x11").get_i32(0) as *mut libc::stat;
                // 3. Stat file
                let ret;
                unsafe { ret = libc::fstat(fd, buf) };
                // 4. Return 0
                ret as u64
            }
        );
        Syscaller::def(169, "gettimeofday",
            |ctx| {
                // ========= Get Time ========= //
                // 1. Get x10/a0 Reg
                let tv = ctx.get_nreg("x10").get_i32(0) as *mut libc::timeval;
                // 2. Get x11/a1 Reg
                let tz = ctx.get_nreg("x11").get_i32(0) as *mut libc::timezone;
                // 3. Get Time
                let ret;
                unsafe { ret = libc::gettimeofday(tv, tz) };
                // 4. Return 0
                ret as u64
            }
        );
        Syscaller::def(214, "brk",
            |ctx| {
                // ========= Get Time ========= //
                // 1. Get x10/a0 Reg
                let addr = ctx.get_nreg("x10").get_i32(0) as *mut c_void;
                // 2. Get x11/a1 Reg
                let ret;
                unsafe { ret = libc::brk(addr) };
                // 3. Return 0
                ret as u64
            }
        )
    }

    /// Get syscall
    pub fn pool_get(id : usize) -> (&'static str, fn(&IRContext) -> u64) {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow().get(&id).unwrap().clone()
        })
    }

    /// Set syscall
    pub fn pool_set(id : usize, name: &'static str, call: fn(&IRContext) -> u64) {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow_mut().insert(id, (name, call));
        })
    }

    /// Delete syscall
    pub fn pool_del(id : usize) {
        Self::SYSCALL_POOL.with(|pool| {
            pool.borrow_mut().remove(&id);
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
    pub fn pool_is_in(id : usize) -> bool {
        Self::SYSCALL_POOL.with(|pool| pool.borrow().get(&id).is_some())
    }

    /// Pool Info
    pub fn pool_info() -> String {
        let mut info = String::new();
        info.push_str(&format!("Syscalls (Num = {}):\n", Self::pool_size()));
        Self::SYSCALL_POOL.with(|pool| {
            for (id, (name, _)) in pool.borrow().iter() {
                info.push_str(&format!("[{:>4}] {}\n", id, name));
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