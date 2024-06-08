

// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

#[cfg(not(feature = "no-log"))]
use colored::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use crate::core::op::Operand;
use crate::util::log::Span;
use crate::log_info;

// ============================================================================== //
//                              task::TaskProfile
// ============================================================================== //

#[cfg(not(feature = "no-profile"))]
#[derive(Debug, Clone, PartialEq)]
pub struct TaskProfile {

    pub insn_in_cnt: usize,                 // load insn count
    pub insn_out_cnt: usize,                // gen insn count
    pub insn_max_cnt: usize,                // max gen insn count in BB
    pub insn_del_cnt: usize,                // Opt insn and del count
    
    pub code_in_len: usize,                 // load code len
    pub code_out_len: usize,                // gen code len

    pub switch_time: std::time::Duration,   // switch time
    pub exec_time: std::time::Duration,     // exec time
    pub opt_time: std::time::Duration,      // opt time
    pub dump_time: std::time::Duration,     // dump time
}

// ============================================================================== //
//                              task::TaskContextStatus
// ============================================================================== //
#[derive(Debug, Clone, PartialEq)]
pub enum TaskContextStatus {
    /// (SBT/DBT) Idle: Nothing
    Idle,
    /// (SBT/DBT) Load elf and map to memory
    Load,
    /// (SBT/DBT) Fetch code bytes
    Fetch,
    /// (SBT/DBT) Disassembly bytes to insns
    Disas,
    /// (SBT/DBT) Translation Arch to Arch
    Trans,
    /// (SBT/DBT) Optimize: Opt IR Arch
    Opt,
    /// (SBT/DBT) Switch Mode: Save and Wait for next status
    Switch,
    /// (DBT) Execution Arch Block
    Exec,
    /// (SBT/DBT) Dump/Restore code to file
    Dump
}

impl TaskContextStatus {
    #[cfg(not(feature = "no-log"))]
    pub fn to_string(&self) -> String {
        match self {
            TaskContextStatus::Idle => "Idle".white().to_string(),
            TaskContextStatus::Load => "Load".blue().to_string(),
            TaskContextStatus::Fetch => "Fetch".bright_yellow().to_string(),
            TaskContextStatus::Disas => "Disas".yellow().to_string(),
            TaskContextStatus::Trans => "Trans".cyan().to_string(),
            TaskContextStatus::Opt => "Opt".bright_blue().to_string(),
            TaskContextStatus::Switch => "Switch".magenta().to_string(),
            TaskContextStatus::Exec => "Exec".green().to_string(),
            TaskContextStatus::Dump => "Dump".purple().to_string(),
        }
    }



    #[cfg(feature = "no-log")]
    pub fn to_string(&self) -> String {
        match self {
            TaskContextStatus::Idle => "Idle".to_string(),
            TaskContextStatus::Load => "Load".to_string(),
            TaskContextStatus::Fetch => "Fetch".to_string(),
            TaskContextStatus::Disas => "Disas".to_string(),
            TaskContextStatus::Trans => "Trans".to_string(),
            TaskContextStatus::Opt => "Opt".to_string(),
            TaskContextStatus::Switch => "Switch".to_string(),
            TaskContextStatus::Exec => "Exec".to_string(),
            TaskContextStatus::Dump => "Dump".to_string(),
        }
    }
}

// ============================================================================== //
//                              task::TaskContext
// ============================================================================== //

#[derive(Debug, Clone, PartialEq)]
pub struct TaskContext {

    pub status: TaskContextStatus,

    /// Dispatch label table
    pub labels: Rc<RefCell<HashMap<String, Rc<RefCell<Operand>>>>>,

    #[cfg(not(feature = "no-profile"))]
    pub prof: TaskProfile,
}

impl TaskContext {

    pub fn new() -> Self {
        Self {
            status: TaskContextStatus::Idle,
            labels: Rc::new(RefCell::new(HashMap::new())),

            #[cfg(not(feature = "no-profile"))]
            prof: TaskProfile {
                insn_in_cnt: 0,
                insn_out_cnt: 0,
                insn_max_cnt: 0,
                insn_del_cnt: 0,
                code_in_len: 0,
                code_out_len: 0,
                switch_time: std::time::Duration::from_millis(0),
                exec_time: std::time::Duration::from_millis(0),
                opt_time: std::time::Duration::from_millis(0),
                dump_time: std::time::Duration::from_millis(0),
            }
        }
    }


    
    /// Set label
    pub fn set_label(&self, lab: Rc<RefCell<Operand>>) {
        let nick = lab.borrow().label_nick();
        log_info!("set label: `{}`", nick);
        self.labels.borrow_mut().insert(nick, lab);
    }

    /// Get label
    pub fn get_label(&self, nick: String) -> Rc<RefCell<Operand>> {
        self.labels.borrow().get(&nick).unwrap().clone()
    }

    /// del label
    pub fn del_label(&self, nick: String) {
        log_info!("del label: `{}`", nick);
        self.labels.borrow_mut().remove(&nick);
    }
}