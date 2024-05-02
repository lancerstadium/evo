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
use std::fmt;
#[cfg(not(feature = "no-log"))]
use colored::*;

// use crate::util::log::Span;
use crate::ir::val::IRValue;
// use crate::log_error;
use crate::ir::ctx::{IRContext, ArchInfo};
use crate::ir::op::IRInsn;
use crate::log_warning;
use crate::util::log::Span;
use crate::ir::ty::IRType;



// ============================================================================== //
//                              mem::IRThreadStatus
// ============================================================================== //

#[derive(Debug, Clone, PartialEq)]
pub enum IRThreadStatus {
    /// Ready: Thread is ready
    Ready,
    /// Running: Thread is running
    Running,
    /// Blocked: Thread is blocked
    Blocked,
    /// Stopped: Thread is Stopped
    Stopped,
    /// Unknown: Thread status is unknown
    Unknown,
}

impl IRThreadStatus {
    #[cfg(not(feature = "no-log"))]
    pub fn to_string(&self) -> String {
        match self {
            IRThreadStatus::Ready => "Ready".blue().to_string(),
            IRThreadStatus::Running => "Running".green().to_string(),
            IRThreadStatus::Blocked => "Blocked".yellow().to_string(),
            IRThreadStatus::Stopped => "Stopped".red().to_string(),
            IRThreadStatus::Unknown => "Unknown".purple().to_string(),
        }
    }

    #[cfg(feature = "no-log")]
    pub fn to_string(&self) -> String {
        match self {
            IRThreadStatus::Ready => String::from("Ready"),
            IRThreadStatus::Running => String::from("Running"),
            IRThreadStatus::Blocked => String::from("Blocked"),
            IRThreadStatus::Stopped => String::from("Stopped"),
            IRThreadStatus::Unknown => String::from("Unknown"),
        }
    }
}

impl fmt::Display for IRThreadStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


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
    pub registers: Vec<Rc<RefCell<IRValue>>>,
    /// `stack`: Stack, Store stack value
    pub stack: Vec<Rc<RefCell<IRValue>>>,
    /// Dispatch label table
    pub labels : Rc<RefCell<Vec<String>>>,
    /// Thread status
    pub status : IRThreadStatus,
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
            registers: Vec::new(),
            stack: Vec::new(),
            labels: Rc::new(RefCell::new(Vec::new())),
            status: IRThreadStatus::Ready,
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
        let thread_ptr = Self::pool_last().1;
        thread_ptr.borrow_mut().id = Self::IR_THREAD_POOL.with(|pool| pool.borrow().len() - 1);
        let mut init_regs = Vec::new();
        if IRContext::is_32() {
            init_regs = (0..IRContext::REG_NUM).map(|_| Rc::new(RefCell::new(IRValue::i32(0)))).collect::<Vec<_>>();
        } else if IRContext::is_64() {
            init_regs = (0..IRContext::REG_NUM).map(|_| Rc::new(RefCell::new(IRValue::i64(0)))).collect::<Vec<_>>();
        }
        thread_ptr.borrow_mut().registers.extend(init_regs);
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

    /// Info all thread
    #[cfg(not(feature = "no-log"))]
    pub fn pool_info_tbl() -> String {
        let mut info = String::new();
        info.push_str(&format!("┌─────┬─────┬──────┬───────────┬─────────┐\n"));
        info.push_str(&format!("│ TID │ PID │ Regs │   Stack   │ TStatus │\n"));
        Self::IR_THREAD_POOL.with(|pool| {
            let borrowed_pool = pool.borrow();
            for i in 0..borrowed_pool.len() {
                let thread = borrowed_pool[i].borrow();
                let stk_pc = thread.stack_scale() as f64 / IRContext::STACK_SIZE as f64 * 100.0;
                let stk_fmt = format!("{:.2}%", stk_pc);
                info.push_str(&format!("├─────┼─────┼──────┼───────────┼─────────┤\n"));
                info.push_str(&format!("│ {:^3} │ {:^3} │ {:>3}  │ {:^9} │ {:^16} │\n", 
                    thread.id, thread.proc_id, 
                    thread.reg_num(), stk_fmt, 
                    thread.status.to_string()
                ));
            }
        });
        info.push_str(&format!("└─────┴─────┴──────┴───────────┴─────────┘\n"));
        info
    }

    // ================= IRThread.reg ==================== //

    /// set reg value by index
    pub fn set_reg(&self, index: usize, value: IRValue) {
        let regs = self.registers.clone();
        regs[index].borrow_mut().change(value);
    }

    /// get reg value by index
    pub fn get_reg(&self, index: usize) -> IRValue {
        let val = self.registers[index].borrow().clone();
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
            self.registers[index].replace(IRValue::u32(0));
        } else if IRContext::is_64() {
            self.registers[index].replace(IRValue::u64(0));
        }
    }

    /// get reg info
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
            info.push_str(&format!("- {} -> {}\n", reg.info(), self.registers[i].borrow().clone().bin(0, -1, true)));
        }
        info
    }

    /// get reg num
    pub fn reg_num(&self) -> usize {
        self.registers.len()
    }

    /// Get pc: ABI x2 is pc
    pub fn get_pc(&self) -> IRValue {
        self.get_nreg("x2")
    }

    /// Set pc: ABI x2 is pc
    pub fn set_pc(&self, value: IRValue) {
        self.set_nreg("x2", value)
    }

    // ================= IRThread.stark ================== //

    /// stack push value
    pub fn stack_push(&mut self, value: IRValue) {
        self.stack.push(Rc::new(RefCell::new(value)));
    }

    /// stack pop value
    pub fn stack_pop(&mut self) -> IRValue {
        self.stack.pop().unwrap().borrow().clone()
    }

    /// stack clear
    pub fn stack_clr(&mut self) {
        self.stack.clear();
    }

    /// stack last value
    pub fn stack_last(&self) -> IRValue {
        self.stack.last().unwrap().borrow().clone()
    }

    /// stack len: stack vec size
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }

    /// stack scale: stack scale size
    pub fn stack_scale(&self) -> usize {
        // Sum all IRValue scale
        self.stack.iter().map(|v| v.borrow().scale_sum()).sum()
    }

    // ================= IRThread.proc =================== //

    /// find father process
    pub fn proc(&self) -> Rc<RefCell<IRProcess>> {
        IRProcess::pool_get(self.proc_id)
    }

    // ================= IRThread.status ================= //

    /// get status
    pub fn status(&self) -> &IRThreadStatus {
        &self.status
    }

    /// set status
    pub fn set_status(&mut self, status: IRThreadStatus) {
        self.status = status;
    }

}


impl default::Default for IRThread {
    fn default() -> Self {
        Self {
            id: 0,
            proc_id: 0,
            registers: Vec::new(),
            stack: Vec::new(),
            labels: Rc::new(RefCell::new(Vec::new())),
            status: IRThreadStatus::Ready,
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
    pub code_segment: Rc<RefCell<Vec<IRValue>>>,
    /// `mem_segment`: Data Segment
    pub mem_segment: Rc<RefCell<IRValue>>,
    /// `threads_id`: Threads ID
    pub threads_id: Rc<RefCell<Vec<usize>>>,
    /// `cur_thread`: current thread
    pub cur_thread: Rc<RefCell<IRThread>>
}


impl IRProcess {

    thread_local! {
        pub static IR_PROCESS_POOL: Rc<RefCell<Vec<Rc<RefCell<IRProcess>>>>> = Rc::new(RefCell::new(Vec::new()));

    }
    // ================= IRProcess.ctl =================== //

    /// Init `IRProcess`
    pub fn init(name: &'static str) -> Rc<RefCell<IRProcess>> {
        let proc : IRProcess;
        proc = Self {
            id:0,
            name,
            code_segment: Rc::new(RefCell::new(Vec::new())),
            mem_segment: Rc::new(RefCell::new(IRValue::default())),
            threads_id: Rc::new(RefCell::new(Vec::new())),
            cur_thread: Rc::new(RefCell::new(IRThread::default())),
        };
        // Store in process pool
        let idx = Self::pool_push(proc);
        let val = Self::pool_get(idx);
        val.borrow_mut().apply_thread();
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
        if IRContext::is_32() {
            proc.mem_segment.borrow_mut().set_type(IRType::array(IRType::u32(), IRContext::MEM_SIZE / 4));
        } else if IRContext::is_64() {
            proc.mem_segment.borrow_mut().set_type(IRType::array(IRType::u64(), IRContext::MEM_SIZE / 8));
        }
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

    /// Info all process
    #[cfg(not(feature = "no-log"))]
    pub fn pool_info_tbl() -> String {
        let mut info = String::new();
        info.push_str(&format!("┌─────┬──────────┬────────────────┬────────┬─────────┬───────────┬─────────┐\n"));
        info.push_str(&format!("│ PID │   name   │      TIDs      │  Code  │   Mem   │   Stack   │ TStatus │\n"));
        Self::IR_PROCESS_POOL.with(|pool| {
            let borrowed_pool = pool.borrow();
            for i in 0..borrowed_pool.len() {
                let proc = borrowed_pool[i].borrow();
                let thread = proc.cur_thread.borrow();
                // name first 8 char
                let name_fmt = proc.name.chars().take(8).collect::<String>();
                // make Vec<usize> to string: Get first 5 TID nums, other TIDs will be replaced by ...
                let mut tid_fmt = proc.threads_id.borrow().iter().map(
                    |tid| tid.to_string()
                ).take(5).collect::<Vec<String>>().join(",");
                if proc.threads_id.borrow().len() > 5 {
                    tid_fmt.push_str("...");
                }
                let code_fmt = Self::usize_to_string(proc.code_segment.borrow().len());
                let mem_fmt = Self::usize_to_string(proc.mem_segment.borrow().scale_sum() / 8);
                let stack_pc = thread.stack_scale() as f64 / IRContext::STACK_SIZE as f64 * 100.0;
                let stack_fmt = format!("{:.2}%", stack_pc);
                let status_fmt = thread.status().to_string();
                info.push_str(&format!("├─────┼──────────┼────────────────┼────────┼─────────┼───────────┼─────────┤\n"));
                info.push_str(&format!("│ {:^3} │ {:^8} │ {:^14} │ {:^6} │ {:^7} │ {:^9} │ {:^16} │\n", 
                    pool.borrow()[i].borrow().id, 
                    name_fmt,
                    tid_fmt,
                    code_fmt,
                    mem_fmt,
                    stack_fmt,
                    status_fmt
                ));
            }
        });
        info.push_str(&format!("└─────┴──────────┴────────────────┴────────┴─────────┴───────────┴─────────┘\n"));
        info
    }

    /// usize to store String: B, KB, MB, GB
    pub fn usize_to_string(size: usize) -> String {
        let mut size = size;
        let mut unit = "B";
        if size >= 1024 && size < 1024 * 1024 {
            size /= 1024;
            unit = "KB";
        } else if size >= 1024 * 1024 && size < 1024 * 1024 * 1024 {
            size /= 1024 * 1024;
            unit = "MB";
        } else if size >= 1024 * 1024 * 1024 && size < 1024 * 1024 * 1024 * 1024 {
            size /= 1024 * 1024 * 1024;
            unit = "GB";
        }
        format!("{}{}", size, unit)
    }


    // ================= IRProcess.get =================== //

    /// Get info
    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Process {} info: \n", self.id));
        info.push_str(&format!("- name: {}\n- code seg size: {}\n- mem seg size: {} / {}\n- threads: {:?}\n- thread id: {}\n- thread reg num: {}\n- thread stack size: {} / {}\n", 
            self.name, self.code_segment.borrow().len(), 
            self.mem_segment.borrow().scale_sum() / 8, IRContext::MEM_SIZE,
            self.threads_id.borrow().clone(), 
            self.cur_thread.borrow().id, self.cur_thread.borrow().reg_num(), 
            self.cur_thread.borrow().stack_scale(), IRContext::STACK_SIZE));
        info
    }

    // ================= IRProcess.code ================== //

    /// Push code
    pub fn push_code(&self, code: IRValue) {
        let mut code_seg = self.code_segment.borrow_mut().clone();
        code_seg.push(code);
    }

    /// Pop code
    pub fn pop_code(&self) -> IRValue {
        let mut code_seg = self.code_segment.borrow_mut();
        code_seg.pop().unwrap().clone()
    }

    // ================= IRProcess.mem =================== //

    /// Write mem value: by 32 or 64-bit / index
    pub fn write_mem(&self, index: usize, value: IRValue) {
        let idx :usize;
        if IRContext::is_64() {
            idx = index * 8;
        } else {
            idx = index * 4;
        }
        self.mem_segment.borrow_mut().set(idx, value);
    }

    /// Read mem value: by 32 or 64-bit / index (num=[1-16])
    pub fn read_mem(&self, index: usize, num: usize) -> IRValue {
        let mut num = num;
        if num <= 0 {
            log_warning!("Read Mem: num: {} <= 0", num);
            num = 1;
        } else if num > 16 {
            log_warning!("Read Mem: num: {} > 16", num);
            num = 16;
        }
        let idx :usize;
        let scale :usize;
        if IRContext::is_64() {
            idx = index * 8;
            scale = 64;
        } else {
            idx = index * 4;
            scale = 32;
        }
        self.mem_segment.borrow().get(idx, scale * num)
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

    /// New thread: Use a pure new thread, If you wanr a new thread fork from current, use`fork_thread`
    pub fn new_thread(&self) -> Rc<RefCell<IRThread>> {
        let thread_id = IRThread::init(self.id);
        self.threads_id.borrow_mut().push(thread_id);
        IRThread::pool_get(thread_id)
    }

    /// apply thread: at thread vec[0]
    pub fn apply_thread(&mut self) -> Rc<RefCell<IRThread>> {
        let thread_id = self.threads_id.borrow()[0];
        self.cur_thread = IRThread::pool_get(thread_id);
        self.cur_thread.borrow_mut().set_status(IRThreadStatus::Running);
        self.cur_thread.clone()
    }

    /// set thread by index
    pub fn set_thread(&mut self, index: usize) {
        if index >= self.threads_id.borrow().len() {
            log_warning!("Set Thread index: {} >= {}", index, self.threads_id.borrow().len());
            return;
        }
        let thread_id = self.threads_id.borrow()[index];
        self.cur_thread = IRThread::pool_get(thread_id);
        self.cur_thread.borrow_mut().set_status(IRThreadStatus::Running);
    }

    /// fork thread from current: copy thread reg and stack, set thread
    pub fn fork_thread(&mut self) -> Rc<RefCell<IRThread>> {
        // 1. Get new thread
        let new_thread = self.new_thread();
        // 2. Copy reg and stack
        new_thread.borrow_mut().registers = self.cur_thread.borrow().registers.clone();
        new_thread.borrow_mut().stack = self.cur_thread.borrow().stack.clone();
        // 3. Set Status
        self.cur_thread.borrow_mut().set_status(IRThreadStatus::Ready);
        // 4. swap threads_id vec[0] to new thread
        let temp_id = self.threads_id.borrow()[0];
        let new_id = new_thread.borrow().id;
        self.threads_id.borrow_mut()[0] = new_id;
        self.threads_id.borrow_mut().pop();
        self.threads_id.borrow_mut().push(temp_id);
        // 5. Set thread Running
        self.apply_thread();
        new_thread
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

    /// Set reg pc
    pub fn set_pc(&self, value: IRValue) {
        self.cur_thread.borrow_mut().set_pc(value)
    }

    /// Get reg pc
    pub fn get_pc(&self) -> IRValue {
        self.cur_thread.borrow().get_pc()
    }


    // ================= IRProcess.status ================ //

    /// set status
    pub fn set_status(&self, status: IRThreadStatus) {
        self.cur_thread.borrow_mut().set_status(status)
    }

    /// get status
    pub fn status(&self) -> IRThreadStatus {
        self.cur_thread.borrow().status().clone()
    }
}

