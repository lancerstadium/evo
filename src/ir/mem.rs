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

// use crate::util::log::Span;
use crate::ir::val::IRValue;
// use crate::log_error;
use crate::ir::ctx::{IRContext, ArchInfo};
use crate::ir::op::IRInsn;



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
    /// self Thread ID
    pub id: usize,
    /// `registers`: Register File, Store register value
    pub registers: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// `stack`: Stack, Store stack value
    pub stack: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// Dispatch label table
    pub labels : Rc<RefCell<Vec<String>>>,
}


impl IRThread {

    thread_local! {
        /// `IR_THREAD_POOL`: Thread Pool     [  Thread ID  |  (Registers, Stack)  ]
        pub static IR_THREAD_POOL: Rc<RefCell<Vec<Rc<RefCell<IRThread>>>>> = Rc::new(RefCell::new(Vec::new()));
    }

    // ================= IRThread.ctl ==================== //

    pub fn init(proc_id: usize) -> usize {
        let thread : IRThread;
        thread = Self {
            id : 0,
            proc_id,
            registers: Rc::new(RefCell::new(Vec::new())),
            stack: Rc::new(RefCell::new(Vec::new())),
            labels: Rc::new(RefCell::new(Vec::new())),
        };
        // Store in thread pool
        let idx = Self::pool_push(thread);
        idx
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
        let mut thread = Self::pool_last().1.borrow_mut().clone();
        thread.id = Self::IR_THREAD_POOL.with(|pool| pool.borrow().len() - 1);
        let mut init_regs = Vec::new();
        // let mut init_stack = Vec::new();
        if IRContext::is_32() {
            init_regs = (0..IRContext::REG_NUM).map(|_| Rc::new(RefCell::new(IRValue::i32(0)))).collect::<Vec<_>>();
            // init_stack = (0..IRContext::STACK_SIZE / 4).map(|_| Rc::new(RefCell::new(IRValue::u32(0)))).collect::<Vec<_>>();
        } else if IRContext::is_64() {
            init_regs = (0..IRContext::REG_NUM).map(|_| Rc::new(RefCell::new(IRValue::i64(0)))).collect::<Vec<_>>();
            // init_stack = (0..IRContext::STACK_SIZE / 8).map(|_| Rc::new(RefCell::new(IRValue::u64(0)))).collect::<Vec<_>>();
        }
        thread.registers.borrow_mut().extend(init_regs);
        // thread.stack.borrow_mut().extend(init_stack);
        Self::pool_last().0
    }

    /// Pop thread from pool
    pub fn pool_pop() -> Option<Rc<RefCell<IRThread>>> {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow_mut().pop())
    }

    /// Get last index
    pub fn pool_last() -> (usize, Rc<RefCell<IRThread>>) {
        let idx = Self::IR_THREAD_POOL.with(|pool| pool.borrow().len() - 1);
        (idx, Self::pool_get(idx))
    }

    /// Get pool size
    pub fn pool_size() -> usize {
        Self::IR_THREAD_POOL.with(|pool| pool.borrow().len())
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
        let regs = self.registers.borrow_mut().clone();
        regs[index].borrow_mut().change(value);
        // println!("vec: {:?}", regs);
    }

    /// get reg value by index
    pub fn get_reg(&self, index: usize) -> IRValue {
        let val = self.registers.borrow_mut()[index].borrow().clone();
        val
    }

    /// set reg value by name
    pub fn set_nreg(&self, name: &'static str, value: IRValue) {
        let index = IRInsn::reg_pool_nget(name).borrow().clone().val().get_byte(0) as usize;
        self.set_reg(index, value)
    }

    /// get reg value by name
    pub fn get_nreg(&self, name: &'static str) -> IRValue {
        let index = IRInsn::reg_pool_nget(name).borrow().clone().val().get_byte(0) as usize;
        self.get_reg(index)
    }

    /// set reg zero
    pub fn set_reg_zero(&self, index: usize) {
        if IRContext::is_32() {
            self.registers.borrow_mut()[index].replace(IRValue::u32(0));
        } else if IRContext::is_64() {
            self.registers.borrow_mut()[index].replace(IRValue::u64(0));
        }
    }

    pub fn reg_info(&self, start: usize, num: i32) -> String {
        let mut info = String::new();
        info.push_str(&format!("Registers in thread {} (Num = {}):\n", self.id, IRInsn::reg_pool_size()));
        let ss = start;
        let mut ee = start + num as usize;
        if num < 0 || ee > IRInsn::reg_pool_size() {
            ee = IRInsn::reg_pool_size();
        }
        for i in ss..ee {
            let reg = IRInsn::reg_pool_get(i).borrow().clone();
            info.push_str(&format!("- {} -> {}\n", reg.info(), self.registers.borrow()[i].borrow().clone().bin(0, -1, true)));
        }
        info
    }

    pub fn reg_num(&self) -> usize {
        self.registers.borrow().len()
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

    /// stack last value
    pub fn stack_last(&self) -> IRValue {
        self.stack.borrow().last().unwrap().borrow().clone()
    }

    /// stack len: stack vec size
    pub fn stack_len(&self) -> usize {
        self.stack.borrow().len()
    }

    /// stack scale: stack scale size
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
            id: 0,
            proc_id: 0,
            registers: Rc::new(RefCell::new(Vec::new())),
            stack: Rc::new(RefCell::new(Vec::new())),
            labels: Rc::new(RefCell::new(Vec::new())),
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
    /// `id`: Process ID
    pub id: usize,
    /// `name`: Process Name
    pub name: &'static str,
    /// `code_segment`: Code Segment
    pub code_segment: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// `data_segment`: Data Segment
    pub data_segment: Rc<RefCell<Vec<Rc<RefCell<IRValue>>>>>,
    /// `threads_id`: Threads ID
    pub threads_id: Rc<RefCell<Vec<usize>>>,
    /// `cur_thread`: current thread
    pub cur_thread: Rc<RefCell<IRThread>>,
}


impl IRProcess {

    thread_local! {
        pub static IR_PROCESS_POOL: Rc<RefCell<Vec<Rc<RefCell<IRProcess>>>>> = Rc::new(RefCell::new(Vec::new()));

    }
    // ================= IRProcess.ctl =================== //

    /// Init `IRProcess`
    pub fn init(name: &'static str) -> Self {
        let proc : IRProcess;
        proc = Self {
            id:0,
            name,
            code_segment: Rc::new(RefCell::new(Vec::new())),
            data_segment: Rc::new(RefCell::new(Vec::new())),
            threads_id: Rc::new(RefCell::new(Vec::new())),
            cur_thread: Rc::new(RefCell::new(IRThread::default())),
        };
        // Store in process pool
        let idx = Self::pool_push(proc);
        let mut val = Self::pool_get(idx).borrow_mut().clone();
        val.set_thread();
        val
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
        let mut proc = Self::pool_last().1.borrow_mut().clone();
        proc.id = Self::IR_PROCESS_POOL.with(|pool| pool.borrow().len() - 1);
        let thread_id = IRThread::init(proc.id);
        proc.threads_id.borrow_mut().push(thread_id);
        proc.cur_thread = IRThread::pool_get(thread_id);
        Self::pool_last().0
    }

    /// Get last index
    pub fn pool_last() -> (usize, Rc<RefCell<IRProcess>>) {
        let idx = Self::IR_PROCESS_POOL.with(|pool| pool.borrow().len() - 1);
        (idx, Self::pool_get(idx))
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

    /// Get pool size
    pub fn pool_size() -> usize {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow().len())
    }

    /// Check is in pool
    pub fn pool_is_in(index: usize) -> bool {
        Self::IR_PROCESS_POOL.with(|pool| pool.borrow_mut().get(index).is_some())
    }


    // ================= IRProcess.get =================== //

    /// Get info
    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Process {} info: \n", self.id));
        info.push_str(&format!("- name: {}\n- code seg size: {}\n- threads: {:?}\n- thread id: {}\n- thread reg num: {}\n- thread stack size: {} / {}\n", 
            self.name, self.code_segment.borrow().len(), self.threads_id.borrow().clone(), 
            self.cur_thread.borrow().id, self.cur_thread.borrow().reg_num(), 
            self.cur_thread.borrow().stack_scale(), IRContext::STACK_SIZE));
        info
    }

    // ================= IRProcess.code ================== //

    /// Push code
    pub fn push_code(&self, code: IRValue) {
        let mut code_seg = self.code_segment.borrow_mut();
        let code_ptr = Rc::new(RefCell::new(code));
        code_seg.push(code_ptr);
    }

    /// Pop code
    pub fn pop_code(&self) -> IRValue {
        let mut code_seg = self.code_segment.borrow_mut();
        code_seg.pop().unwrap().borrow().clone()
    }

    // ================= IRProcess.thread ================ //

    /// Get threads
    pub fn threads(&self) -> Vec<Rc<RefCell<IRThread>>> {
        let mut threads = Vec::new();
        for id in self.threads_id.borrow().iter() {
            threads.push(IRThread::pool_get(*id));
        }
        threads
    }

    /// New thread
    pub fn new_thread(&self) -> Rc<RefCell<IRThread>> {
        let thread_id = IRThread::init(self.id);
        self.threads_id.borrow_mut().push(thread_id);
        IRThread::pool_get(thread_id)
    }

    /// set thread
    pub fn set_thread(&mut self) -> Rc<RefCell<IRThread>> {
        let thread_id = self.threads_id.borrow()[0];
        self.cur_thread = IRThread::pool_get(thread_id);
        self.cur_thread.clone()
    }


    // ================= IRProcess.reg =================== //

    /// set reg value by index
    pub fn set_reg(&self, index: usize, value: IRValue) {
        self.cur_thread.borrow_mut().set_reg(index, value)
    }

    /// get reg value by index
    pub fn get_reg(&self, index: usize) -> IRValue {
        self.cur_thread.borrow().get_reg(index)
    }

    /// set reg value by name
    pub fn set_nreg(&self, name: &'static str, value: IRValue) {
        self.cur_thread.borrow_mut().set_nreg(name, value)
    }

    /// get reg value by name
    pub fn get_nreg(&self, name: &'static str) -> IRValue {
        self.cur_thread.borrow().get_nreg(name)
    }

    /// get reg info
    pub fn reg_info(&self, start: usize, num: i32) -> String {
        self.cur_thread.borrow().reg_info(start, num)
    }

    /// get reg num
    pub fn reg_num(&self) -> usize {
        self.cur_thread.borrow().reg_num()
    }

}


impl default::Default for IRProcess {
    /// Set default function for `IRProcess`: id = 0, name = "main"
    fn default() -> Self {
        Self::init("main")
    }
}