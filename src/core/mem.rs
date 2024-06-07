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
use std::collections::HashMap;
use std::default;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;

use std::fs::File;
use std::io::Read;
use std::ffi::CStr;
use std::ptr;
use goblin::elf::{Elf, header::Header};
use libc::{c_void, MAP_PRIVATE, MAP_ANONYMOUS, PROT_READ, PROT_WRITE, MAP_FAILED, mmap, munmap};
// use std::path::PathBuf;
// use std::os::unix::io::AsRawFd;

#[cfg(not(feature = "no-log"))]
use colored::*;
use crate::arch::evo::def::EVO_ARCH;
use crate::arch::info::Arch;
// use crate::util::log::Span;
use crate::core::val::Value;
// use crate::log_error;
use crate::log_error;
use crate::log_info;
use crate::log_warning;
use crate::util::log::Span;
use crate::core::ty::Types;

use super::insn::RegFile;
use super::op::Operand;



// ============================================================================== //
//                              mem::MemoryTool
// ============================================================================== //

#[derive(Debug, Clone)]
pub struct MemoryTool {


}

impl MemoryTool {

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

    /// Elf class info
    pub fn elf_class_info(e_class: u8) -> String {
        match e_class {
            1 => "ELF32",
            2 => "ELF64",
            _ => "Unknown",
        }.to_string()
    }

    /// Elf data info
    pub fn elf_data_info(e_data: u8) -> String {
        match e_data {
            1 => "2's complement, little endian",
            2 => "2's complement, big endian",
            _ => "Unknown",
        }.to_string()
    }

    /// Elf Version info
    pub fn elf_version_info(e_version: u8) -> String {
        match e_version {
            0 => "Invalid",
            1 => "Current",
            _ => "Unknown",
        }.to_string()
    }

    /// Elf osabi info
    pub fn elf_osabi_info(e_osabi: u8) -> String {
        match e_osabi {
            0 => "Unix System V",
            3 => "Linux",
            6 => "Sun Solaris",
            8 => "IBM AIX",
            _ => "Unknown",
        }.to_string()
    }

    /// Elf machine info
    pub fn elf_machine_info(e_machine: u16) -> String {
        match e_machine {
            243 => "riscv",
            _ => "Unknown",
        }.to_string()
    }

    /// Print elf header
    fn elf_header_info(header: &Header) {
        println!("ELF header:");
        println!("  Magic:   {}", header.e_ident.to_vec().iter().map(|x| format!("{:02x}", x)).collect::<Vec<String>>().join(" "));
        println!("  Class:                             {} ({})", header.e_ident[4], MemoryTool::elf_class_info(header.e_ident[4]));
        println!("  Data:                              {} ({})", header.e_ident[5], MemoryTool::elf_data_info(header.e_ident[5]));
        println!("  Version:                           {} ({})", header.e_ident[6], MemoryTool::elf_version_info(header.e_ident[6]));
        println!("  OS/ABI:                            {} ({})", header.e_ident[7], MemoryTool::elf_osabi_info(header.e_ident[7]));
        println!("  ABI Version:                       {}", header.e_ident[8]);
        println!("  Type:                              {}", header.e_type);
        println!("  Machine:                           {} ({})", header.e_machine, MemoryTool::elf_machine_info(header.e_machine));
        println!("  Version:                           {}", header.e_version);
        println!("  Entry point address:               0x{:x} ({})", header.e_entry, header.e_entry);
        println!("  Start of program headers:          {} (bytes into file)", header.e_phoff);
        println!("  Start of section headers:          {} (bytes into file)", header.e_shoff);
        println!("  Flags:                             0x{:x}", header.e_flags);
        println!("  Size of this header:               {} (bytes)", header.e_ehsize);
        println!("  Size of program headers:           {} (bytes)", header.e_phentsize);
        println!("  Number of program headers:         {}", header.e_phnum);
        println!("  Size of section headers:           {} (bytes)", header.e_shentsize);
        println!("  Number of section headers:         {}", header.e_shnum);
        println!("  Section header string table index: {}", header.e_shstrndx);
    }


    /// Elf loader
    pub fn elf_load(path: &str) -> Option<(*mut u8, usize, usize)> {
        // 1. Get ELF File
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
    
        // 2. Parse ELF
        let elf = Elf::parse(&buffer).unwrap();

        // print ELF header info: file name, file size, entry point, architecture, endiness, bits, platform
        Self::elf_header_info(&elf.header);
    
        // 3. Get entry point
        let entry_point = elf.header.e_entry as usize;
    
        // 4. Calculate total size of all segments
        let total_size: usize = elf.program_headers.iter().filter(|phdr| phdr.p_type == goblin::elf::program_header::PT_LOAD).map(|phdr| phdr.p_filesz as usize).sum();
    
        // 5. Map the entire file to a continuous memory region
        let file_ptr = unsafe {
            mmap(
                ptr::null_mut(),
                total_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
    
        // 6. Check mmap
        if file_ptr == MAP_FAILED {
            let err_msg = unsafe {
                let err_code = *libc::__errno_location();
                let c_err_msg = libc::strerror(err_code);
                CStr::from_ptr(c_err_msg).to_string_lossy().into_owned()
            };
            log_error!("mmap failed: {}", err_msg);
            return None;
        }
    
        // 7. Copy file contents to mapped memory
        let mut offset = 0;
        for phdr in elf.program_headers {
            if phdr.p_type == goblin::elf::program_header::PT_LOAD {
                let segment_size = phdr.p_filesz as usize;
                let segment_offset = phdr.p_offset as usize;
                let segment_ptr = (file_ptr as usize + offset) as *mut u8;
                println!("Segment: {} - {} = {} bytes, vaddr: 0x{:x}", segment_offset, segment_offset + segment_size, segment_size, phdr.p_vaddr);
                // Find .text segment and modify entry point index to it
                unsafe {
                    ptr::copy(buffer.as_ptr().add(segment_offset), segment_ptr, segment_size);
                }
                offset += segment_size;
            }
        }
    
        Some((file_ptr as *mut u8, total_size, entry_point))
    }
    

    /// Elf unloader
    pub fn elf_unload(load_segs: (*mut u8, usize, usize)) {
        let (ptr, size, _) = load_segs;
        unsafe { munmap(ptr as *mut c_void, size) };
    }

    /// Seg to Value
    pub fn seg_to_val(seg: (*mut u8, usize, usize)) -> Value {
        let (ptr, size, _) = seg;
        let val_refcell: RefCell<Vec<u8>> = unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, size);
            RefCell::new(Vec::from(slice))
        };
        Value::array_u8(val_refcell)
    }
}


#[cfg(test)]
mod memtool_test {

    use crate::arch::riscv::def::RISCV32_ARCH;
    use crate::core::cpu::CPUState;
    use crate::core::insn::Instruction;
    use super::*;

    #[test]
    fn elf_test() {
        CPUState::init(&RISCV32_ARCH, &RISCV32_ARCH, None, None, None);
        let seg = MemoryTool::elf_load("/home/lexer/item/evo-rs/test/hello.elf").unwrap();
        println!("segs: {:?}", seg);
        let val = MemoryTool::seg_to_val(seg);
        // println!("val: {}", val.hex(0, -1, false));

        // Read Mem
        for i in 0..20 {
            let insn_val = val.get( seg.2 + i * 4, 32);
            let insn = Instruction::decode(&RISCV32_ARCH, insn_val.clone());
            println!("Mem: {}  -> Dec: {}", insn_val.bin(0, -1, true), insn);
        }

        MemoryTool::elf_unload(seg);
    }
}


// ============================================================================== //
//                              mem::CPUThreadStatus
// ============================================================================== //

#[derive(Debug, Clone, PartialEq)]
pub enum CPUThreadStatus {
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

impl CPUThreadStatus {
    #[cfg(not(feature = "no-log"))]
    pub fn to_string(&self) -> String {
        match self {
            CPUThreadStatus::Ready => "Ready".blue().to_string(),
            CPUThreadStatus::Running => "Running".green().to_string(),
            CPUThreadStatus::Blocked => "Blocked".yellow().to_string(),
            CPUThreadStatus::Stopped => "Stopped".red().to_string(),
            CPUThreadStatus::Unknown => "Unknown".purple().to_string(),
        }
    }

    #[cfg(feature = "no-log")]
    pub fn to_string(&self) -> String {
        match self {
            CPUThreadStatus::Ready => String::from("Ready"),
            CPUThreadStatus::Running => String::from("Running"),
            CPUThreadStatus::Blocked => String::from("Blocked"),
            CPUThreadStatus::Stopped => String::from("Stopped"),
            CPUThreadStatus::Unknown => String::from("Unknown"),
        }
    }
}

impl fmt::Display for CPUThreadStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


// ============================================================================== //
//                              mem::CPUThread
// ============================================================================== //

/// 1. `threads`: Thread Handle (Local Thread): contains (registers, stack)
/// 2. `registers`: Register File, Store register value
/// 3. `stack`: Stack, Store stack value
#[derive(Debug, Clone)]
pub struct CPUThread {
    /// `arch`
    pub arch: &'static Arch,
    /// `stack_size`
    pub stack_size: usize,
    /// `proc_id`: Process ID
    pub proc_id: usize,
    /// self Thread ID
    pub id: usize,
    /// pc step: PC step size
    pub pc_step : usize,
    /// `registers`: Register File, Store register value
    pub registers: Vec<Rc<RefCell<Value>>>,
    /// `stack`: Stack, Store stack value
    pub stack: Vec<Rc<RefCell<Value>>>,
    /// Dispatch label table
    pub labels : Rc<RefCell<HashMap<String, Rc<RefCell<Operand>>>>>,
    /// Thread status
    pub status : CPUThreadStatus,
}


impl CPUThread {

    thread_local! {
        /// `CPU_THREAD_POOL`: Thread Pool     [  Thread ID  |  (Registers, Stack)  ]
        pub static CPU_THREAD_POOL: Rc<RefCell<Vec<Rc<RefCell<CPUThread>>>>> = Rc::new(RefCell::new(Vec::new()));
    }

    // ================= CPUThread.ctl ==================== //

    pub fn init(proc: &CPUProcess) -> usize {
        let thread : CPUThread;
        let proc_id = proc.id;
        let stack_size = proc.stack_size;
        let arch = proc.arch;
        thread = Self {
            arch,
            stack_size,
            id : 0,
            proc_id,
            pc_step : 4,
            registers: Vec::new(),
            stack: Vec::new(),
            labels: Rc::new(RefCell::new(HashMap::new())),
            status: CPUThreadStatus::Ready,
        };
        // Store in thread pool
        let idx = Self::pool_push(thread);
        idx
    }


    // ================= CPUThread.pool =================== //

    /// Get thread by index
    pub fn pool_get(index: usize) -> Rc<RefCell<CPUThread>> {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut()[index].clone())
    }

    /// Set thread by index
    pub fn pool_set(index: usize , thread: CPUThread) {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut()[index] = Rc::new(RefCell::new(thread)));
    }

    /// Push thread into pool and return index
    pub fn pool_push(thread: CPUThread) -> usize {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(thread))));
        let thread_ptr = Self::pool_last().1;
        thread_ptr.borrow_mut().id = Self::CPU_THREAD_POOL.with(|pool| pool.borrow().len() - 1);
        let mut init_regs = Vec::new();
        let reg_num = thread_ptr.borrow().arch.reg_num;
        if thread_ptr.borrow().arch.mode.is_32bit() {
            init_regs = (0..reg_num).map(|_| Rc::new(RefCell::new(Value::i32(0)))).collect::<Vec<_>>();
        } else if thread_ptr.borrow().arch.mode.is_64bit() {
            init_regs = (0..reg_num).map(|_| Rc::new(RefCell::new(Value::i64(0)))).collect::<Vec<_>>();
        }
        thread_ptr.borrow_mut().registers.extend(init_regs);
        Self::pool_last().0
    }

    /// Pop thread from pool
    pub fn pool_pop() -> Option<Rc<RefCell<CPUThread>>> {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut().pop())
    }

    /// Get last index
    pub fn pool_last() -> (usize, Rc<RefCell<CPUThread>>) {
        let idx = Self::CPU_THREAD_POOL.with(|pool| pool.borrow().len() - 1);
        (idx, Self::pool_get(idx))
    }

    /// Get pool size
    pub fn pool_size() -> usize {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow().len())
    }

    /// Clear Thread Pool
    pub fn pool_clr() {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Delete thread by index
    pub fn pool_del(index: usize) {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut().remove(index));
    }

    /// Check is in pool
    pub fn pool_is_in(index: usize) -> bool {
        Self::CPU_THREAD_POOL.with(|pool| pool.borrow_mut().get(index).is_some())
    }

    /// Info all thread
    #[cfg(not(feature = "no-log"))]
    pub fn pool_info_tbl() -> String {
        let mut info = String::new();
        info.push_str(&format!("┌─────┬─────┬──────┬───────────┬─────────┐\n"));
        info.push_str(&format!("│ TID │ PID │ Regs │   Stack   │ TStatus │\n"));
        Self::CPU_THREAD_POOL.with(|pool| {
            let borrowed_pool = pool.borrow();
            for i in 0..borrowed_pool.len() {
                let thread = borrowed_pool[i].borrow();
                let stack_size = thread.stack_size;
                let stk_pc = thread.stack_scale() as f64 / stack_size as f64 * 100.0;
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

    // ================= CPUThread.reg ==================== //

    /// set reg value by index
    pub fn set_reg(&self, index: usize, value: Value) {
        let regs = self.registers.clone();
        regs[index].borrow_mut().change(value);
    }

    /// get reg value by index
    pub fn get_reg(&self, index: usize) -> Value {
        let val = self.registers[index].borrow().clone();
        val
    }

    /// set reg value by name
    pub fn set_nreg(&self, name: &'static str, value: Value) {
        let index = RegFile::reg_poolr_nget(self.arch, name).borrow().clone().val().get_byte(0) as usize;
        self.set_reg(index, value)
    }

    /// get reg value scale offset by name
    pub fn get_nreg_all(&self, name: &'static str) -> (Value, usize, usize) {
        let index = RegFile::reg_poolr_nget(self.arch, name).borrow().clone().val().get_byte(0) as usize;
        let val = self.get_reg(index);
        let scale = RegFile::reg_poolr_nget(self.arch, name).borrow().clone().reg_scale();
        let offset = RegFile::reg_poolr_nget(self.arch, name).borrow().clone().reg_offset();
        val.bound(offset, scale);
        (val, offset, scale)
    }

    /// get reg value by name
    pub fn get_nreg(&self, name: &'static str) -> Value{
        let index = RegFile::reg_poolr_nget(self.arch, name).borrow().clone().val().get_byte(0) as usize;
        self.get_reg(index)
    }

    /// set reg zero
    pub fn set_reg_zero(&self, index: usize) {
        if self.arch.mode.is_32bit() {
            self.registers[index].replace(Value::i32(0));
        } else if self.arch.mode.is_64bit() {
            self.registers[index].replace(Value::i64(0));
        }
    }

    /// get reg info
    pub fn reg_info(&self, start: usize, num: i32) -> String {
        let mut info = String::new();
        info.push_str(&format!("Registers in thread {} (Num = {}):\n", self.id, self.reg_num()));
        let ss = start;
        let mut ee = start + num as usize;
        if num < 0 || ee > self.reg_num() {
            ee = self.reg_num();
        }
        for i in ss..ee {
            let reg = RegFile::reg_poolr_get(self.arch, i).borrow().clone();
            info.push_str(&format!("- {} -> {}\n", reg.info(), self.registers[i].borrow().clone().bin(0, -1, true)));
        }
        info
    }

    /// get reg num
    pub fn reg_num(&self) -> usize {
        self.registers.len()
    }

    /// Get pc: ABI pc
    pub fn get_pc(&self) -> Value {
        self.get_nreg_all("pc").0
    }

    /// Set pc: ABI pc
    pub fn set_pc(&self, value: Value) {
        self.set_nreg("pc", value)
    }

    /// Get next pc
    pub fn get_pc_next(&self) -> Value {
        if self.arch.mode.is_32bit() {
            Value::i32(self.get_pc().get_i32(0) + self.pc_step as i32)
        } else if self.arch.mode.is_64bit() {
            Value::i64(self.get_pc().get_i64(0) + self.pc_step as i64)
        } else {
            unreachable!()
        }
    }

    /// Set next pc: set pc_step
    pub fn set_pc_next(&mut self, step: Value) {
        if self.arch.mode.is_32bit() {
            self.pc_step = step.get_i32(0) as usize;
            self.set_pc(Value::i32(self.get_pc().get_i32(0) + self.pc_step as i32));
        } else if self.arch.mode.is_64bit() {
            self.pc_step = step.get_i64(0) as usize;
            self.set_pc(Value::i64(self.get_pc().get_i64(0) + step.get_i64(0) as i64));
        } else {
            unreachable!()
        }
    }


    /// Set label
    pub fn set_label(&self, lab: Rc<RefCell<Operand>>) {
        let nick = lab.borrow().label_nick();
        log_info!("[ TID_{:<2}] set label: `{}`", self.id, nick);
        self.labels.borrow_mut().insert(nick, lab);
    }

    /// Get label
    pub fn get_label(&self, nick: String) -> Rc<RefCell<Operand>> {
        self.labels.borrow().get(&nick).unwrap().clone()
    }

    /// del label
    pub fn del_label(&self, nick: String) {
        log_info!("[ TID_{:<2}] del label: `{}`", self.id, nick);
        self.labels.borrow_mut().remove(&nick);
    }


    // ================= CPUThread.stark ================== //

    /// stack push value
    pub fn stack_push(&mut self, value: Value) {
        self.stack.push(Rc::new(RefCell::new(value)));
    }

    /// stack pop value
    pub fn stack_pop(&mut self) -> Value {
        self.stack.pop().unwrap().borrow().clone()
    }

    /// stack clear
    pub fn stack_clr(&mut self) {
        self.stack.clear();
    }

    /// stack last value
    pub fn stack_last(&self) -> Value {
        self.stack.last().unwrap().borrow().clone()
    }

    /// stack len: stack vec size
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }

    /// stack scale: stack scale size
    pub fn stack_scale(&self) -> usize {
        // Sum all Value scale
        self.stack.iter().map(|v| v.borrow().scale_sum()).sum()
    }

    // ================= CPUThread.proc =================== //

    /// find father process
    pub fn proc(&self) -> Rc<RefCell<CPUProcess>> {
        CPUProcess::pool_get(self.proc_id)
    }

    // ================= CPUThread.status ================= //

    /// get status
    pub fn status(&self) -> &CPUThreadStatus {
        &self.status
    }

    /// set status
    pub fn set_status(&mut self, status: CPUThreadStatus) {
        self.status = status;
    }

}


impl default::Default for CPUThread {
    fn default() -> Self {
        Self {
            arch: &EVO_ARCH,
            stack_size: 0,
            id: 0,
            proc_id: 0,
            pc_step : 0,
            registers: Vec::new(),
            stack: Vec::new(),
            labels: Rc::new(RefCell::new(HashMap::new())),
            status: CPUThreadStatus::Ready,
        }
    }
}

impl cmp::PartialEq for CPUThread {
    fn eq(&self, other: &Self) -> bool {
        self.proc_id == other.proc_id
    }
}




// ============================================================================== //
//                              mem::CPUProcess
// ============================================================================== //

/// `CPUProcess`: Process Handle (Global Thread): contains (Code Segment, Data Segment, threads)
#[derive(Debug, Clone, PartialEq)]
pub struct CPUProcess {
    /// `arch`: CPU Process arch
    pub arch: &'static Arch,
    /// `base_addr`
    pub base_addr: usize,
    /// `mem_size`:
    pub mem_size: usize,
    /// `stack_size`:
    pub stack_size: usize,
    /// `id`: Process ID
    pub id: usize,
    /// `name`: Process Name
    pub name: String,
    /// `code_segment`: Code Segment
    pub code_segment: Rc<RefCell<Vec<Value>>>,
    /// `mem_segment`: Data Segment
    pub mem_segment: Rc<RefCell<Value>>,
    /// `threads_id`: Threads ID
    pub threads_id: Rc<RefCell<Vec<usize>>>,
    /// `cur_thread`: current thread
    pub cur_thread: Rc<RefCell<CPUThread>>
}


impl CPUProcess {

    thread_local! {
        pub static CPU_PROCESS_POOL: Rc<RefCell<Vec<Rc<RefCell<CPUProcess>>>>> = Rc::new(RefCell::new(Vec::new()));

    }
    // ================= CPUProcess.ctl =================== //

    /// Init `CPUProcess`
    pub fn init(arch: &'static Arch, base_addr: usize, mem_size: usize, stack_size: usize) -> Rc<RefCell<CPUProcess>> {
        let proc : CPUProcess;
        proc = Self {
            arch,
            base_addr,
            mem_size,
            stack_size,
            id:0,
            name: arch.to_string(),
            code_segment: Rc::new(RefCell::new(Vec::new())),
            mem_segment: Rc::new(RefCell::new(Value::default())),
            threads_id: Rc::new(RefCell::new(Vec::new())),
            cur_thread: Rc::new(RefCell::new(CPUThread::default())),
        };
        // Store in process pool
        let idx = Self::pool_push(proc);
        let val = Self::pool_get(idx);
        val.borrow_mut().apply_thread();
        val
    }


    // ================= CPUProcess.pool ================== //

    /// Get process by index
    pub fn pool_get(index: usize) -> Rc<RefCell<CPUProcess>> {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow_mut()[index].clone())
    }

    /// Set process by index
    pub fn pool_set(index: usize , process: CPUProcess) {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow_mut()[index] = Rc::new(RefCell::new(process)));
    }

    /// Push process into pool and return index
    pub fn pool_push(process: CPUProcess) -> usize {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow_mut().push(Rc::new(RefCell::new(process))));
        let proc = Self::pool_last().1;
        proc.borrow_mut().id = Self::CPU_PROCESS_POOL.with(|pool| pool.borrow().len() - 1);
        let thread_id = CPUThread::init(&*proc.borrow());
        proc.borrow_mut().threads_id.borrow_mut().push(thread_id);
        proc.borrow_mut().cur_thread = CPUThread::pool_get(thread_id);
        let mem_size = proc.borrow().mem_size;
        if proc.borrow().arch.mode.is_32bit() {
            proc.borrow_mut().mem_segment.borrow_mut().set_type(Types::array(Types::u32(), mem_size / 4));
        } else if proc.borrow().arch.mode.is_64bit() {
            proc.borrow_mut().mem_segment.borrow_mut().set_type(Types::array(Types::u64(), mem_size / 8));
        }
        Self::pool_last().0
    }

    /// Get last index
    pub fn pool_last() -> (usize, Rc<RefCell<CPUProcess>>) {
        let idx = Self::CPU_PROCESS_POOL.with(|pool| pool.borrow().len() - 1);
        (idx, Self::pool_get(idx))
    }

    /// Delete process by index
    pub fn pool_del(index: usize) {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow_mut().remove(index));
    }

    /// Clear Process Pool
    pub fn pool_clr() {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Get process by name
    pub fn pool_nget(name: &'static str) -> Rc<RefCell<CPUProcess>> {
        Self::CPU_PROCESS_POOL.with(|pool| {
            pool.borrow_mut().iter().find(|p| p.borrow_mut().name == name).unwrap().clone()
        })
    }

    /// Set process by name
    pub fn pool_nset(name: &'static str, process: CPUProcess) {
        Self::CPU_PROCESS_POOL.with(|pool| {
            pool.borrow_mut().iter().find(|p| p.borrow_mut().name == name).unwrap().replace(process)
        });
    }

    /// Get proc idx
    pub fn pool_idx(name: &'static str) -> usize {
        Self::CPU_PROCESS_POOL.with(|pool| {
            pool.borrow_mut().iter().position(|p| p.borrow_mut().name == name).unwrap()
        })
    }

    /// Get pool size
    pub fn pool_size() -> usize {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow().len())
    }

    /// Check is in pool
    pub fn pool_is_in(index: usize) -> bool {
        Self::CPU_PROCESS_POOL.with(|pool| pool.borrow_mut().get(index).is_some())
    }

    /// Info all process
    #[cfg(not(feature = "no-log"))]
    pub fn pool_info_tbl() -> String {
        let mut info = String::new();
        info.push_str(&format!("┌─────┬────────────┬────────────────┬────────┬─────────┬───────────┬─────────┐\n"));
        info.push_str(&format!("│ PID │    name    │      TIDs      │  Code  │   Mem   │   Stack   │ TStatus │\n"));
        Self::CPU_PROCESS_POOL.with(|pool| {
            let borrowed_pool = pool.borrow();
            for i in 0..borrowed_pool.len() {
                let proc = borrowed_pool[i].borrow();
                let thread = proc.cur_thread.borrow();
                // name first 10 char
                let name_fmt = proc.name.chars().take(10).collect::<String>();
                // make Vec<usize> to string: Get first 5 TID nums, other TIDs will be replaced by ...
                let mut tid_fmt = proc.threads_id.borrow().iter().map(
                    |tid| tid.to_string()
                ).take(5).collect::<Vec<String>>().join(",");
                if proc.threads_id.borrow().len() > 5 {
                    tid_fmt.push_str("...");
                }
                let stack_size = proc.stack_size;
                let code_fmt = MemoryTool::usize_to_string(proc.code_segment.borrow().len());
                let mem_fmt = MemoryTool::usize_to_string(proc.mem_segment.borrow().scale_sum() / 8);
                let stack_pc = thread.stack_scale() as f64 / stack_size as f64 * 100.0;
                let stack_fmt = format!("{:.2}%", stack_pc);
                let status_fmt = thread.status().to_string();
                info.push_str(&format!("├─────┼────────────┼────────────────┼────────┼─────────┼───────────┼─────────┤\n"));
                info.push_str(&format!("│ {:^3} │ {:^10} │ {:^14} │ {:^6} │ {:^7} │ {:^9} │ {:^16} │\n", 
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
        info.push_str(&format!("└─────┴────────────┴────────────────┴────────┴─────────┴───────────┴─────────┘\n"));
        info
    }


    // ================= CPUProcess.get =================== //

    /// Get info
    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Process {} info: \n", self.id));
        info.push_str(&format!("- name: {}\n- code seg size: {}\n- mem seg size: {} / {}\n- threads: {:?}\n- thread id: {}\n- thread reg num: {}\n- thread stack size: {} / {}\n", 
            self.name, self.code_segment.borrow().len(), 
            self.mem_segment.borrow().scale_sum() / 8, self.mem_size,
            self.threads_id.borrow().clone(), 
            self.cur_thread.borrow().id, self.cur_thread.borrow().reg_num(), 
            self.cur_thread.borrow().stack_scale(), self.stack_size));
        info
    }

    // ================= CPUProcess.code ================== //

    /// Push code
    pub fn push_code(&self, code: Value) {
        let mut code_seg = self.code_segment.borrow_mut().clone();
        code_seg.push(code);
    }

    /// Pop code
    pub fn pop_code(&self) -> Value {
        let mut code_seg = self.code_segment.borrow_mut();
        code_seg.pop().unwrap().clone()
    }

    // ================= CPUProcess.mem =================== //

    /// Write mem value: by 32 or 64-bit / index
    pub fn mem_write(&self, index: usize, value: Value) {
        let is_64 = self.arch.mode.is_64bit();
        let idx :usize;
        if is_64 {
            idx = index * 8;
        } else {
            idx = index * 4;
        }
        self.mem_segment.borrow_mut().set(idx, value);
    }

    /// Read mem value: by 32 or 64-bit / index (num=[1-16])
    pub fn mem_read(&self, index: usize, num: usize) -> Value {
        let is_64 = self.arch.mode.is_64bit();
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
        if is_64 {
            idx = index * 8;
            scale = 64;
        } else {
            idx = index * 4;
            scale = 32;
        }
        self.mem_segment.borrow().get(idx, scale * num)
    }

    pub fn mem_opread(&self, addr_base: usize, addr_idx: usize, addr_scale: u8, addr_disp: i32, data_scale: usize) -> Value {
        let is_64 = self.arch.mode.is_64bit();
        let base_val;
        let idx_val;
        if is_64 {
            base_val = self.get_reg(addr_base).get_i64(0);
            idx_val = self.get_reg(addr_idx).get_i64(0);
        } else {
            base_val = self.get_reg(addr_base).get_i32(0) as i64;
            idx_val = self.get_reg(addr_idx).get_i32(0) as i64;
        }
        let index = (base_val + idx_val * addr_scale as i64 + addr_disp as i64) as usize;
        self.mem_segment.borrow().get(index, data_scale * 8)
    }

    pub fn mem_opwrite(&self, addr_base: usize, addr_idx: usize, addr_scale: u8, addr_disp: i32, data_scale: usize, value: Value) {
        let is_64 = self.arch.mode.is_64bit();
        let base_val;
        let idx_val;
        if is_64 {
            base_val = self.get_reg(addr_base).get_i64(0);
            idx_val = self.get_reg(addr_idx).get_i64(0);
        } else {
            base_val = self.get_reg(addr_base).get_i32(0) as i64;
            idx_val = self.get_reg(addr_idx).get_i32(0) as i64;
        }
        let index = (base_val + idx_val * addr_scale as i64 + addr_disp as i64) as usize;
        // get new value by scale
        let mut new_value = value.clone();
        new_value.set_scale(data_scale * 8);
        self.mem_segment.borrow_mut().set(index, new_value);
    }

    // ================= CPUProcess.thread ================ //

    /// Get threads
    pub fn threads(&self) -> Vec<Rc<RefCell<CPUThread>>> {
        let mut threads = Vec::new();
        for id in self.threads_id.borrow().iter() {
            threads.push(CPUThread::pool_get(*id));
        }
        threads
    }

    /// New thread: Use a pure new thread, If you wanr a new thread fork from current, use`fork_thread`
    pub fn new_thread(&self) -> Rc<RefCell<CPUThread>> {
        let thread_id = CPUThread::init(self);
        self.threads_id.borrow_mut().push(thread_id);
        CPUThread::pool_get(thread_id)
    }

    /// apply thread: at thread vec[0]
    pub fn apply_thread(&mut self) -> Rc<RefCell<CPUThread>> {
        let thread_id = self.threads_id.borrow()[0];
        self.cur_thread = CPUThread::pool_get(thread_id);
        self.cur_thread.borrow_mut().set_status(CPUThreadStatus::Running);
        self.cur_thread.clone()
    }

    /// set thread by index
    pub fn set_thread(&mut self, index: usize) {
        if index >= self.threads_id.borrow().len() {
            log_warning!("Set Thread index: {} >= {}", index, self.threads_id.borrow().len());
            return;
        }
        let thread_id = self.threads_id.borrow()[index];
        self.cur_thread = CPUThread::pool_get(thread_id);
        self.cur_thread.borrow_mut().set_status(CPUThreadStatus::Running);
    }

    /// fork thread from current: copy thread reg and stack, set thread
    pub fn fork_thread(&mut self) -> Rc<RefCell<CPUThread>> {
        // 1. Get new thread
        let new_thread = self.new_thread();
        // 2. Copy reg and stack
        new_thread.borrow_mut().registers = self.cur_thread.borrow().registers.clone();
        new_thread.borrow_mut().stack = self.cur_thread.borrow().stack.clone();
        // 3. Set Status
        self.cur_thread.borrow_mut().set_status(CPUThreadStatus::Ready);
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


    // ================= CPUProcess.reg =================== //

    /// set reg value by index
    pub fn set_reg(&self, index: usize, value: Value) {
        self.cur_thread.borrow_mut().set_reg(index, value)
    }

    /// get reg value by index
    pub fn get_reg(&self, index: usize) -> Value {
        self.cur_thread.borrow().get_reg(index)
    }

    /// set reg value by name
    pub fn set_nreg(&self, name: &'static str, value: Value) {
        self.cur_thread.borrow_mut().set_nreg(name, value)
    }

    /// get reg (val, offset, scale) by name
    pub fn get_nreg_all(&self, name: &'static str) -> (Value, usize, usize) {
        self.cur_thread.borrow().get_nreg_all(name)
    }

    /// get reg val by name
    pub fn get_nreg(&self, name: &'static str) -> Value {
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
    pub fn set_pc(&self, value: Value) {
        self.cur_thread.borrow_mut().set_pc(value)
    }

    /// Get reg pc
    pub fn get_pc(&self) -> Value {
        self.cur_thread.borrow().get_pc()
    }

    /// Get pc step
    pub fn get_pc_step(&self) -> usize {
        self.cur_thread.borrow().pc_step
    }

    /// Set pc step
    pub fn set_pc_step(&self, step: usize) {
        self.cur_thread.borrow_mut().pc_step = step
    }


    pub fn get_pc_next(&self) -> Value {
        self.cur_thread.borrow().get_pc_next()
    }

    pub fn set_pc_next(&self, step: Value) {
        self.cur_thread.borrow_mut().set_pc_next(step)
    }


    pub fn set_label(&self, lab: Rc<RefCell<Operand>>) {
        self.cur_thread.borrow_mut().set_label(lab)
    }

    pub fn get_label(&self, nick: String) -> Rc<RefCell<Operand>> {
        self.cur_thread.borrow().get_label(nick)
    }

    pub fn del_label(&self, nick: String) {
        self.cur_thread.borrow_mut().del_label(nick)
    }

    // ================= CPUProcess.stack ================= //

    /// stack pop
    pub fn stack_pop(&self) -> Value {
        self.cur_thread.borrow_mut().stack_pop()
    }

    /// stack push
    pub fn stack_push(&self, value: Value) {
        self.cur_thread.borrow_mut().stack_push(value)
    }

    // ================= CPUProcess.status ================ //

    /// set status
    pub fn set_status(&self, status: CPUThreadStatus) {
        self.cur_thread.borrow_mut().set_status(status)
    }

    /// get status
    pub fn status(&self) -> CPUThreadStatus {
        self.cur_thread.borrow().status().clone()
    }
}
