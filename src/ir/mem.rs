//! `evo::ir::mem` module
//! ### Process and Thread
//! 
//! - Here are the process and thread in the `evo-ir` architecture
//! 
//! ```
//!           ┌──────────────┬────────────────┬──────┐
//!  Process  │ Code Segment │  Data Segment  │ Heap │
//!  ───────  ├──────────────┴──┬─────────────┴──────┤
//!  Threads  │      Stack      │     Registers      │
//!           └─────────────────┴────────────────────┘
//!
//! ```


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::cmp;
use std::default;
use std::rc::Rc;
use std::cell::RefCell;

use crate::util::log::Span;
use crate::ir::val::IRValue;
use crate::log_error;
use crate::ir::ctx::{IRContext, ArchInfo};



// ============================================================================== //
//                              mem::IRThread
// ============================================================================== //



/// 1. `threads`: Thread Handle (Local Thread): contains (registers, stack)
/// 2. `registers`: Register File, Store register value
/// 3. `stack`: Stack, Store stack value
#[derive(Debug, Clone)]
pub struct IRThread {
    /// `proc_id`: Process ID
    pub proc_id: usize,
    /// `registers`: Register File, Store register value
    pub registers: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// `stack`: Stack, Store stack value
    pub stack: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
}


impl IRThread {

    thread_local! {
        /// `IR_THREAD_POOL`: Thread Pool     [  Thread ID  |  (Registers, Stack)  ]
        pub static IR_THREAD_POOL: Rc<RefCell<Vec<Rc<RefCell<IRThread>>>>> = Rc::new(RefCell::new(Vec::new()));
    }

    // ================= IRThread.ctl ==================== //

    pub fn init(proc_id: usize) -> usize {
        let thread : IRThread;
        if IRContext::is_32() {
            thread = Self {
                proc_id,
                registers: Rc::new(RefCell::new(vec![Rc::new(RefCell::new(IRValue::u32(0))); IRContext::REG_NUM])),
                stack: Rc::new(RefCell::new(Vec::new())),
            };
            // Store in thread pool
            Self::pool_push(thread)
        } else if IRContext::is_64() {
            thread = Self {
                proc_id,
                registers: Rc::new(RefCell::new(vec![Rc::new(RefCell::new(IRValue::u64(0))); IRContext::REG_NUM])),
                stack: Rc::new(RefCell::new(Vec::new())),
            };
            // Store in thread pool
            Self::pool_push(thread)
        } else {
            log_error!("Unsupported architecture");
            0
        }
    }

    // ================= IRThread.pool =================== //

    /// Get thread by index
    pub fn pool_get(index: usize) -> Rc<RefCell<IRThread>> {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut()[index].clone())
    }

    /// Set thread by index
    pub fn pool_set(index: usize , thread: IRThread) {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut()[index] = Rc::new(RefCell::new(thread)));
    }

    /// Push thread into pool and return index
    pub fn pool_push(thread: IRThread) -> usize {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(thread))));
        Self::IR_THREAD_POOL.with(|pool| pool.borrow().len() - 1)
    }

    /// Pop thread from pool
    pub fn pool_pop() -> Option<Rc<RefCell<IRThread>>> {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut().pop())
    }

    /// Clear Thread Pool
    pub fn pool_clr() {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Delete thread by index
    pub fn pool_del(index: usize) {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut().remove(index));
    }

    /// Check is in pool
    pub fn pool_is_in(index: usize) -> bool {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut().get(index).is_some())
    }

    // ================= IRThread.reg ==================== //

    /// set reg value by index
    pub fn set_reg(&self, index: usize, value: IRValue) {
        self.registers.borrow_mut()[index].replace(value);
    }

    /// get reg value by index
    pub fn get_reg(&self, index: usize) -> IRValue {
        self.registers.borrow()[index].borrow().clone()
    }

    /// set reg zero
    pub fn set_reg_zero(&self, index: usize) {
        if IRContext::is_32() {
            self.registers.borrow_mut()[index].replace(IRValue::u32(0));
        } else if IRContext::is_64() {
            self.registers.borrow_mut()[index].replace(IRValue::u64(0));
        }
    }

    // ================= IRThread.stark ================== //

    /// stack push value
    pub fn stack_push(&self, value: IRValue) {
        self.stack.borrow_mut().push(Rc::new(RefCell::new(value)));
    }

    /// stack pop value
    pub fn stack_pop(&self) -> IRValue {
        self.stack.borrow_mut().pop().unwrap().borrow().clone()
    }

    /// stack clear
    pub fn stack_clr(&self) {
        self.stack.borrow_mut().clear();
    }

    /// stack size
    pub fn stack_size(&self) -> usize {
        self.stack.borrow().len()
    }

    /// stack scale
    pub fn stack_scale(&self) -> usize {
        // Sum all IRValue scale
        self.stack.borrow().iter().map(|v| v.borrow().scale_sum()).sum()
    }

    // ================= IRThread.proc =================== //

    /// find father process
    pub fn proc(&self) -> Rc<RefCell<IRProcess>> {
        IRProcess::pool_get(self.proc_id)
    }

}


impl default::Default for IRThread {
    fn default() -> Self {
        Self {
            proc_id: 0,
            registers: Rc::new(RefCell::new(Vec::new())),
            stack: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl cmp::PartialEq for IRThread {
    fn eq(&self, other: &Self) -> bool {
        self.proc_id == other.proc_id
    }
}

// ============================================================================== //
//                              mem::IRProcess
// ============================================================================== //

/// `IRProcess`: Process Handle (Global Thread): contains (Code Segment, Data Segment, threads)
#[derive(Debug, Clone, PartialEq)]
pub struct IRProcess {
    /// `name`: Process Name
    pub name: &'static str,
    /// `code_segment`: Code Segment
    pub code_segment: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// `data_segment`: Data Segment
    pub data_segment: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// `threads_id`: Threads ID
    pub threads_id: Rc<RefCell<Rc<RefCell<Vec<usize>>>>>,
}


impl IRProcess {

    thread_local! {
        pub static IR_PROCESS_POOL: Rc<RefCell<Vec<Rc<RefCell<IRProcess>>>>> = Rc::new(RefCell::new(Vec::new()));

    }
    // ================= IRProcess.ctl =================== //

    /// Init `IRProcess`
    pub fn init(name: &'static str) -> Self {
        let proc : IRProcess;
        if IRContext::is_32() {
            proc = Self {
                name,
                code_segment: Rc::new(RefCell::new(vec![Rc::new(RefCell::new(IRValue::u32(0))); IRContext::MEM_SIZE / 4])),
                data_segment: Rc::new(RefCell::new(Vec::new())),
                threads_id: Rc::new(RefCell::new(Rc::new(RefCell::new(Vec::new())))),
            };
            // Store in process pool
            let idx = Self::pool_push(proc);
            Self::pool_get(idx).borrow().clone()
        } else if IRContext::is_64() {
            proc = Self {
                name,
                code_segment: Rc::new(RefCell::new(vec![Rc::new(RefCell::new(IRValue::u64(0))); IRContext::MEM_SIZE / 8])),
                data_segment: Rc::new(RefCell::new(Vec::new())),
                threads_id: Rc::new(RefCell::new(Rc::new(RefCell::new(Vec::new())))),
            };
            // Store in process pool
            let idx = Self::pool_push(proc);
            Self::pool_get(idx).borrow().clone()
        } else {
            log_error!("Unsupported architecture");
            Self::default()
        }
    }


    // ================= IRProcess.pool ================== //

    /// Get process by index
    pub fn pool_get(index: usize) -> Rc<RefCell<IRProcess>> {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut()[index].clone())
    }

    /// Set process by index
    pub fn pool_set(index: usize , process: IRProcess) {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut()[index] = Rc::new(RefCell::new(process)));
    }

    /// Push process into pool and return index
    pub fn pool_push(process: IRProcess) -> usize {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(process))));
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow().len() - 1)
    }

    /// Delete process by index
    pub fn pool_del(index: usize) {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut().remove(index));
    }

    /// Clear Process Pool
    pub fn pool_clr() {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Get process by name
    pub fn pool_nget(name: &'static str) -> Rc<RefCell<IRProcess>> {
        Self::IR_PROCESS_POOL.with(|pool| {
            pool.borrow_mut().iter().find(|p| p.borrow_mut().name == name).unwrap().clone()
        })
    }

    /// Set process by name
    pub fn pool_nset(name: &'static str, process: IRProcess) {
        Self::IR_PROCESS_POOL.with(|pool| {
            pool.borrow_mut().iter().find(|p| p.borrow_mut().name == name).unwrap().replace(process)
        });
    }

    /// Get proc idx
    pub fn pool_idx(name: &'static str) -> usize {
        Self::IR_PROCESS_POOL.with(|pool| {
            pool.borrow_mut().iter().position(|p| p.borrow_mut().name == name).unwrap()
        })
    }

    /// Check is in pool
    pub fn pool_is_in(index: usize) -> bool {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut().get(index).is_some())
    }


    // ================= IRProcess.get =================== //

    /// Get name
    pub fn name(&self) -> &'static str {
        self.name
    }


    // ================= IRProcess.thread ================ //

    /// Get threads
    pub fn threads(&self) -> Vec<Rc<RefCell<IRThread>>> {
        let mut threads = Vec::new();
        for id in self.threads_id.borrow_mut().borrow().iter() {
            threads.push(IRThread::pool_get(*id));
        }
        threads
    }

    /// New thread
    pub fn new_thread(&self) -> Rc<RefCell<IRThread>> {
        let thread_id = IRThread::init(self.idx());
        self.threads_id.borrow_mut().borrow_mut().push(thread_id);
        IRThread::pool_get(thread_id)
    }

    /// Get idx
    pub fn idx(&self) -> usize {
        Self::pool_idx(self.name())
    }


    // ================= IRProcess.reg =================== //


}


impl default::Default for IRProcess {
    /// Set default function for `IRProcess`: id = 0, name = "main"
    fn default() -> Self {
        Self::init("main")
    }
}