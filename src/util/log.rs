//! `evo::util::log`
//! Use to log



// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::fmt::{self, Arguments};
use std::cell::RefCell;
use std::path::PathBuf;



/// `log`: Use `evo::log`, show color and write to file.
#[cfg(not(feature = "no-log"))]
use std::{fs::File, io::BufRead, io::BufReader, io::Result as IoResult};
#[cfg(not(feature = "no-log"))]
use colored::*;


// ============================================================================== //
//                                 log::Pos
// ============================================================================== //

/// `Pos`: Mark source text position in file.
#[derive(Clone, Copy)]
pub struct Pos {
    /// Line Position
    line: u32,
    /// Column Position
    col: u32,
}

impl Pos {
    /// Creates a new position mark.
    /// - Default `line = 1` and `col = 0`.
    pub fn new() -> Self {
        Self { line: 1, col: 0 }
    }

    /// Updates the line number and column number based on the given character.
    /// - If there is a newline `\n`, make `col = 0` and `self.line ++`.
    pub fn update(&mut self, c: char) {
        match c {
            '\n' => {
                self.col = 0;
                self.line += 1;
            }
            _ => self.col += 1,
        }
    }
}

impl Default for Pos {
    /// Set default function for `Pos`.
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Pos {
    /// Set string for `Pos``.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}


// ============================================================================== //
//                                log::ErrorType
// ============================================================================== //


/// Enum Type of output error info.
/// `no-log`: Don't return error by `evo::log` methods.
#[cfg(feature = "no-log")]
#[derive(Debug)]
pub enum ErrorType {
    /// Normal error: only display.
    Normal(String),
    /// Fatal error: display and stop.
    Fatal(String),
}

/// Enum Type of output error info.
/// `log`: Return error by `evo::log` methods.
#[cfg(not(feature = "no-log"))]
#[derive(Debug)]
pub enum ErrorType {
    /// Normal error: only display.
    Normal,
    /// Fatal error: display and stop.
    Fatal,
}


impl ErrorType {
    /// `no-log`: Return `true` of current error is `Fatal`.
    #[cfg(feature = "no-log")]
    pub fn is_fatal(&self) -> bool {
        matches!(self, ErrorType::Fatal(..))
    }

    /// `log`: Return `true` of current error is `Fatal`.
    #[cfg(not(feature = "no-log"))]
    pub fn is_fatal(&self) -> bool {
        matches!(self, ErrorType::Fatal)
    }
}


impl Default for ErrorType {
    /// `no-log`: Creates a normal error.
    #[cfg(feature = "no-log")]
    fn default() -> Self {
        ErrorType::Normal(String::default())
    }

    /// `log`: Creates a normal error.
    #[cfg(not(feature = "no-log"))]
    fn default() -> Self {
        ErrorType::Normal
    }
}

impl<T> From<ErrorType> for Result<T, ErrorType> {
    /// Creates a result from the given error, value always [`Err`]
    fn from(err: ErrorType) -> Self {
        Err(err)
    }
}


// ============================================================================== //
//                                log::FileType
// ============================================================================== //

/// Enum Type of input file.
pub enum FileType {
    /// File with a path.
    File(PathBuf),
    /// Standaed input file.
    Stdin,
    /// A `Buffer` in memory.
    Buffer,
}

impl fmt::Display for FileType {
    /// Set string for `FileType`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FileType::File(path) => write!(f, "{}", path.display()),
            FileType::Stdin => f.write_str("<stdin>"),
            FileType::Buffer => f.write_str("<buffer>"),
        }
    }
}


// ============================================================================== //
//                              log::GlobalState
// ============================================================================== //

/// `GlobalState`: State of compiler globel info such as: file type, error number ...
struct GlobalState {
    /// Enum Type of file: 
    file: FileType,
    /// Counter of errors
    err_cnt: usize,
    /// Counter of warnings
    warn_cnt: usize,
}



// ============================================================================== //
//                                 log::Span
// ============================================================================== //

/// `Span`: Record source code position in file.
#[derive(Clone, Copy)]
pub struct Span {
    /// start position
    start: Pos,
    /// end position
    end: Pos,
}

/// `get_line_str`(`log`): Get a line str and preprocess it.
/// 1. line -> str (`\t` -> `[space]`)
/// 2. line -> str (`\t` -> `[space]`, `col` update)
/// 3. line -> str (`\t` -> `[space]`, `col1` ~ `col2`)
#[cfg(not(feature = "no-log"))]
macro_rules! get_line_str {
    ($line:expr) => {
        $line
            .map_or("".into(), |r| r.unwrap())
            .replace('\t', &format!("{:w$}", "", w = Span::TAB_WIDTH))
    };
    ($line:expr, $col:expr) => {{
        let line = $line.map_or("".into(), |r| r.unwrap());
        let col = $col as usize;
        let tabs = (&line[..col]).matches('\t').count();
        (
            line.replace('\t', &format!("{:w$}", "", w = Span::TAB_WIDTH)),
            col + tabs * (Span::TAB_WIDTH - 1),
        )
    }};
    ($line:expr, $col1:expr, $col2:expr) => {{
        let line = $line.map_or("".into(), |r| r.unwrap());
        let col1 = $col1 as usize;
        let col2 = $col2 as usize;
        let tabs1 = (&line[..col1]).matches('\t').count();
        let tabs2 = tabs1 + (&line[col1..col2]).matches('\t').count();
        (
            line.replace('\t', &format!("{:w$}", "", w = Span::TAB_WIDTH)),
            col1 + tabs1 * (Span::TAB_WIDTH - 1),
            col2 + tabs2 * (Span::TAB_WIDTH - 1),
        )
    }};
}

impl Span {
    /// `log`: Tab space width: 4.
    #[cfg(not(feature = "no-log"))]
    const TAB_WIDTH: usize = 4;

    thread_local! {
        /// Init GlobalState in `Span.STATE` in static.
        static STATE: RefCell<GlobalState> = RefCell::new(GlobalState {
            file: FileType::Buffer,
            err_cnt: 0,
            warn_cnt: 0,
        });
    }

    // ==================== Span.ctl ===================== //

    /// Create a `Span` from `Pos`.
    pub fn new(pos : Pos) -> Self {
        Self { 
            start : pos, 
            end : pos
        }
    }

    /// Convert to a new `Span`.
    pub fn convert(self, span: Span) -> Self {
        Self {
            start: self.start,
            end: span.end
        }
    }

    /// Get a new span by an end `Pos`.
    pub fn update(self, epos: Pos) -> Self {
        Self {
            start: self.start,
            end: epos
        }
    }

    /// Resets the global state in all spans.
    pub fn reset(file: FileType) {
        Self::STATE.with(|gs| {
            *gs.borrow_mut() = GlobalState {
                file,
                err_cnt: 0,
                warn_cnt: 0,
            }
        });
    }

    // ==================== Span.set ===================== //

    /// Set end position by `Pos`.
    pub fn set_epos(&mut self, epos: Pos) {
        self.end = epos;
    }

    /// Set end position by `Span`.
    pub fn set_espan(&mut self, span: Span) {
        self.end = span.end;
    }


    // ==================== Span.get ===================== //

    /// Gets the number of errors.
    pub fn err_cnt() -> usize {
        Self::STATE.with(|gs| gs.borrow().err_cnt)
    }

    /// Gets the number of warnings.
    pub fn warn_cnt() -> usize {
        Self::STATE.with(|gs| gs.borrow().warn_cnt)
    }


    // ==================== Span.is ====================== //


    /// Check if current span is in same line as given span.
    pub fn is_same_line(&self, span: &Span) -> bool{
        self.end.line == span.start.line
    }

    // ==================== Span.error =================== //

    /// `no-log`: Logs normal error with no span provided.
    #[cfg(feature = "no-log")]
    pub fn log_error(args: Arguments) -> ErrorType {
        // update error number
        Self::STATE.with(|gs| gs.borrow_mut().err_cnt += 1);
        ErrorType::Normal(format!("{}", args))
    }

    /// `log`: Logs normal error with no span provided.
    #[cfg(not(feature = "no-log"))]
    pub fn log_error(args: Arguments) -> ErrorType {
        Self::STATE.with(|gs| {
            // 1. update error counter
            gs.borrow_mut().err_cnt += 1;
            // 2. print message to stderr
            eprintln!("{}: {}", "error".bright_red(), args);
        });
        ErrorType::Normal
    }

    /// `no-log`: Logs fatal error with no span provided.
    #[cfg(feature = "no-log")]
    pub fn log_fatal_error(args: Arguments) -> ErrorType {
        // update error counter
        Self::STATE.with(|gs| gs.borrow_mut().err_cnt += 1);
        ErrorType::Fatal(format!("{}", args))
    }

    /// `log`: Logs fatal error with no span provided.
    #[cfg(not(feature = "no-log"))]
    pub fn log_fatal_error(args: Arguments) -> ErrorType {
        Self::STATE.with(|gs| {
            // 1. update error counter
            gs.borrow_mut().err_cnt += 1;
            // 2. print message to stderr
            eprintln!("{}: {}", "error".bright_red(), args);
        });
        ErrorType::Fatal
    }

    /// `no-log`: Logs normal error message.
    #[cfg(feature = "no-log")]
    pub fn log_error_at(&self, args: Arguments) -> ErrorType {
        Self::log_error(args);
        ErrorType::Normal(self.error_message(args))
    }

    /// `log`: Logs normal error message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_error_at(&self, args: Arguments) -> ErrorType {
        Self::log_error(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::BrightRed));
        ErrorType::Normal
    }

    /// `no-log`: Logs fatal error message.
    #[cfg(feature = "no-log")]
    pub fn log_fatal_error_at(&self, args: Arguments) -> ErrorType {
        Self::log_error(args);
        ErrorType::Fatal(self.error_message(args))
    }

    /// `log`: Logs fatal error message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_fatal_error_at(&self, args: Arguments) -> ErrorType {
        Self::log_error(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::BrightRed));
        ErrorType::Fatal
    }

    // ==================== Span.warning ================= //

    /// `log`: Logs warning with no span provided.
    #[cfg(feature = "no-log")]
    pub fn log_warning(_: Arguments) {
        // update warning number
        Self::STATE.with(|gs| gs.borrow_mut().warn_cnt += 1);
    }

    /// `no-log`: Logs warning with no span provided.
    #[cfg(not(feature = "no-log"))]
    pub fn log_warning(args: Arguments) {
        Self::STATE.with(|gs| {
            // 1. update warning counter
            gs.borrow_mut().warn_cnt += 1;
            // 2. print message to stderr
            eprintln!("{}: {}", "warning".yellow(), args);
        });
    }

    /// `no-log`: Logs warning message.
    #[cfg(feature = "no-log")]
    pub fn log_warning_at(&self, args: Arguments) {
        Self::log_warning(args);
    }

    /// `log`: Logs warning message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_warning_at(&self, args: Arguments) {
        Self::log_warning(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::Yellow));
    }

    // ==================== Span.attention =============== //

    /// `no-log`: Logs attention with no span provided.
    #[cfg(feature = "no-log")]
    pub fn log_attention(_: Arguments) {}

    /// `log`: Logs attention with no span provided.
    #[cfg(not(feature = "no-log"))]
    pub fn log_attention(args: Arguments) {
        eprintln!("{}: {}", "attention".bright_magenta(), args);
    }

    /// `no-log`: Logs attention message.
    #[cfg(feature = "no-log")]
    pub fn log_attention_at(&self, args: Arguments) {
        Self::log_attention(args);
    }

    /// `log`: Logs attention message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_attention_at(&self, args: Arguments) {
        Self::log_attention(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::BrightMagenta));
    }

    // ==================== Span.info ==================== //

    /// `no-log`: Logs info message.
    #[cfg(feature = "no-log")]
    pub fn log_info(_: Arguments) {}

    /// `log`: Logs info message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_info(args: Arguments) {
        eprintln!("{}: {}", "info".bright_green(), args);
    }

    /// `no-log`: Logs info message.
    #[cfg(feature = "no-log")]
    pub fn log_info_at(&self, args: Arguments) {
        Self::log_info(args);
    }

    /// `log`: Logs info message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_info_at(&self, args: Arguments) {
        Self::log_info(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::BrightGreen));
    }

    // ==================== Span.debug =================== //

    /// `no-log`: Logs normal debug message.
    #[cfg(feature = "no-log")]
    pub fn log_debug(_: Arguments) {}

    /// `log`: Logs normal debug message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_debug(args: Arguments) {
        eprintln!("{}: {}", "debug".bright_blue(), args);
    }

    /// `no-log`: Logs normal debug message.
    #[cfg(feature = "no-log")]
    pub fn log_debug_at(&self, args: Arguments) {
        Self::log_debug(args);
    }

    /// `log`: Logs normal debug message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_debug_at(&self, args: Arguments) {
        Self::log_debug(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::BrightBlue));
    }

    // ==================== Span.trace =================== //
    
    /// `log`: Logs normal trace message.
    #[cfg(feature = "no-log")]
    pub fn log_trace(_: Arguments) {}

    /// `no-log`: Logs normal trace message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_trace(args: Arguments) {
        eprintln!("{}: {}", "trace".bright_cyan(), args);
    }

    /// `no-log`: Logs normal trace message.
    #[cfg(feature = "no-log")]
    pub fn log_trace_at(&self, args: Arguments) {
        Self::log_trace(args);
    }

    /// `log`: Logs normal trace message.
    #[cfg(not(feature = "no-log"))]
    pub fn log_trace_at(&self, args: Arguments) {
        Self::log_trace(args);
        Self::STATE.with(|gs| self.log_info_file(&gs.borrow().file, Color::BrightCyan));
    }

    // ==================== Span.global =================== //

    /// `log`: Logs global information (total error/warning number).
    #[cfg(feature = "no-log")]
    pub fn log_global() {}

    /// `no-log`: Logs global information (total error/warning number).
    #[cfg(not(feature = "no-log"))]
    pub fn log_global() {
        Self::STATE.with(|gs| {
            let gs = gs.borrow();
            let mut msg = String::new();
            // 1. error info
            if gs.err_cnt != 0 {
                msg += &format!("{} error", gs.err_cnt);
                if gs.err_cnt > 1 {
                msg += "s";
                }
            }
            // 2. seperator
            if gs.err_cnt != 0 && gs.warn_cnt != 0 {
                msg += " and ";
            }
            // 3. warning info
            if gs.warn_cnt != 0 {
                msg += &format!("{} warning", gs.warn_cnt);
                if gs.warn_cnt > 1 {
                msg += "s";
                }
            }
            // 4. ending
            msg += " emitted";
            eprintln!("{}", msg.bold());
        });
    }


    /// `no-log`: Returns the error message.
    #[cfg(feature = "no-log")]
    fn error_message(&self, args: Arguments) -> String {
        Self::STATE.with(|gs| format!("{}:{}: {}", gs.borrow().file, self.start, args))
    }

    /// Prints the file information.
    #[cfg(not(feature = "no-log"))]
    fn log_info_file(&self, file: &FileType, color: Color) {
        eprintln!("  {} {}:{}", "at".blue(), file, self.start);
        if self.start.col > 0 && self.end.col > 0 {
            if let FileType::File(path) = file {
                // open file and get lines
                let mut lines = BufReader::new(File::open(path).unwrap()).lines();
                if self.start.line == self.end.line {
                    self.log_info_single_line(&mut lines, color);
                } else {
                    self.log_info_multi_line(&mut lines, color);
                }
            }
        }
        eprintln!();
    }

    /// Prints the single line information.
    ///
    /// Used by method `log_info_file`.
    #[cfg(not(feature = "no-log"))]
    fn log_info_single_line<T>(&self, lines: &mut T, color: Color)
    where
        T: Iterator<Item = IoResult<String>>,
    {
        // get some parameters
        let line_num = self.start.line as usize;
        let (line, c1, c2) = get_line_str!(lines.nth(line_num - 1), self.start.col, self.end.col);
        let width = ((line_num + 1) as f32).log10().ceil() as usize;
        let leading = c1 - 1;
        let len = c2 - c1 + 1;
        // print the current line to stderr
        eprintln!("{:w$} {}", "", "|".blue(), w = width);
        eprint!("{} ", format!("{:w$}", line_num, w = width).blue());
        eprintln!("{} {}", "|".blue(), line);
        eprint!("{:w$} {} {:l$}", "", "|".blue(), "", w = width, l = leading);
        eprintln!("{}", format!("{:^>w$}", "", w = len).color(color));
    }

    /// Prints the multi-line information.
    ///
    /// Used by method `log_info_file`.
    #[cfg(not(feature = "no-log"))]
    fn log_info_multi_line<T>(&self, lines: &mut T, color: Color)
    where
        T: Iterator<Item = IoResult<String>>,
    {
        // get some parameters
        let width = ((self.end.line + 1) as f32).log10().ceil() as usize;
        // print the first line to stderr
        let line_num = self.start.line as usize;
        let mut lines = lines.skip(line_num - 1);
        let (line, start) = get_line_str!(lines.next(), self.start.col);
        eprintln!("{:w$} {}", "", "|".blue(), w = width);
        eprint!("{} ", format!("{:w$}", line_num, w = width).blue());
        eprintln!("{}   {}", "|".blue(), line);
        eprint!("{:w$} {}  ", "", "|".blue(), w = width);
        eprintln!("{}", format!("{:_>w$}^", "", w = start).color(color));
        // print the middle lines to stderr
        let mid_lines = (self.end.line - self.start.line) as usize - 1;
        if mid_lines <= 4 {
            for i in 0..mid_lines {
                let line = get_line_str!(lines.next());
                eprint!("{} ", format!("{:w$}", line_num + i + 1, w = width).blue());
                eprintln!("{} {} {}", "|".blue(), "|".color(color), line);
            }
        } else {
            for i in 0..2usize {
                let line = get_line_str!(lines.next());
                eprint!("{} ", format!("{:w$}", line_num + i + 1, w = width).blue());
                eprintln!("{} {} {}", "|".blue(), "|".color(color), line);
            }
            eprint!("{:.>w$} {} {}", "", "|".blue(), "|".color(color), w = width);
            let line = get_line_str!(lines.nth(mid_lines - 3));
            eprint!("{} ", format!("{:w$}", self.end.line - 1, w = width).blue());
            eprintln!("{} {} {}", "|".blue(), "|".color(color), line);
        }
        // print the last line to stderr
        let line_num = self.end.line as usize;
        let (line, end) = get_line_str!(lines.next(), self.end.col);
        eprint!("{} ", format!("{:w$}", line_num, w = width).blue());
        eprintln!("{} {} {}", "|".blue(), "|".color(color), line);
        eprint!("{:w$} {} {}", "", "|".blue(), "|".color(color), w = width);
        eprintln!("{}", format!("{:_>w$}^", "", w = end).color(color));
    }


}

/// Set default function for `Span`.
impl Default for Span {
    fn default() -> Self {
        Self::new(Pos::default())
    }
}

/// Set debug format for `Span`.
impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}


// ============================================================================== //
//                                 Macros
// ============================================================================== //


/// Logs normal error with no span provided.
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)+) => {
        Span::log_error(format_args!($($arg)+))
    };
}

/// Logs fatal error with no span provided.
#[macro_export]
macro_rules! log_fatal_error {
    ($($arg:tt)+) => {
        Span::log_fatal_error(format_args!($($arg)+))
    };
}

/// Logs warning with no span provided.
#[macro_export]
macro_rules! log_warning {
    ($($arg:tt)+) => {
        Span::log_warning(format_args!($($arg)+))
    };
}

/// Logs normal error message.
#[macro_export]
macro_rules! log_error_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_error_at(format_args!($($arg)+))
    };
}

/// Logs fatal error message.
#[macro_export]
macro_rules! log_fatal_error_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_fatal_error_at(format_args!($($arg)+))
    };
}

/// Logs warning message.
#[macro_export]
macro_rules! log_warning_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_warning_at(format_args!($($arg)+))
    };
}

/// Logs error message and returns a result.
#[macro_export]
macro_rules! return_error {
    ($span:expr, $($arg:tt)+) => {
        return $span.log_error_at(format_args!($($arg)+)).into()
    };
}

/// Logs attention with no span provided.
#[macro_export]
macro_rules! log_attention {
    ($($arg:tt)+) => {
        Span::log_attention(format_args!($($arg)+))
    };
}

/// Logs attention message.
#[macro_export]
macro_rules! log_attention_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_attention_at(format_args!($($arg)+))
    };
}

/// Logs info message.
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)+) => {
        Span::log_info(format_args!($($arg)+))
    };
}

/// Logs info message.
#[macro_export]
macro_rules! log_info_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_info_at(format_args!($($arg)+))
    };
}

/// Logs normal debug message.
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)+) => {
        Span::log_debug(format_args!($($arg)+))
    };
}

/// Logs normal debug message.
#[macro_export]
macro_rules! log_debug_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_debug_at(format_args!($($arg)+))
    };
}

/// Logs normal trace message.
#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)+) => {
        Span::log_trace(format_args!($($arg)+))
    };
}

/// Logs normal trace message.
#[macro_export]
macro_rules! log_trace_at {
    ($span:expr, $($arg:tt)+) => {
        $span.log_trace_at(format_args!($($arg)+))
    };
}







// ============================================================================== //
//                                 Tests
// ============================================================================== //


#[cfg(test)]
mod log_test {

    use super::*;


    #[test]
    fn pos_update() {
        let mut pos = Pos::new();
        assert_eq!(format!("{}", pos), "1:0");
        pos.update(' ');
        pos.update(' ');
        assert_eq!(format!("{}", pos), "1:2");
        pos.update('\n');
        assert_eq!(format!("{}", pos), "2:0");
        pos.update('\n');
        pos.update('\n');
        assert_eq!(format!("{}", pos), "4:0");
    }


    #[test]
    fn span_update() {
        let mut pos = Pos::new();
        pos.update(' ');
        let sp1 = Span::new(pos);
        pos.update(' ');
        pos.update(' ');
        let sp2 = sp1.update(pos);
        assert!(sp1.is_same_line(&sp2));
        log_error_at!(sp2, "test error");
        log_warning_at!(sp2, "test warning");
        log_warning_at!(sp2, "test warning 2");
        Span::log_global();
        assert_eq!(format!("{}", sp2.start), "1:1");
        assert_eq!(format!("{}", sp2.end), "1:3");
        let mut sp = Span::new(Pos { line: 10, col: 10 });
        sp.set_epos(Pos { line: 10, col: 15 });
        assert!(!sp2.is_same_line(&sp));
        let sp3 = sp2.convert(sp);
        assert!(sp2.is_same_line(&sp3));
        assert_eq!(format!("{}", sp3.start), "1:1");
        assert_eq!(format!("{}", sp3.end), "10:15");
    }


    #[test]
    fn span_log() {
        log_error_at!(Span::new(Pos::new()), "test error at");
        log_fatal_error_at!(Span::new(Pos::new()), "test fatal error at");
        log_warning_at!(Span::new(Pos::new()), "test warning at");
        log_attention_at!(Span::new(Pos::new()), "test attention at");
        log_info_at!(Span::new(Pos::new()), "test info at");
        log_debug_at!(Span::new(Pos::new()), "test debug at");
        log_trace_at!(Span::new(Pos::new()), "test trace at");

        log_error!("test error");
        log_fatal_error!("test fatal error");
        log_warning!("test warning");
        log_attention!("test attention");
        log_info!("test info");
        log_debug!("test debug");
        log_trace!("test trace");
    }

}