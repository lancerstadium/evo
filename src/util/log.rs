//! `evo::util::log`
//! Use to log



// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::fmt::{self};
use std::cell::RefCell;
use std::fs::File;



// ============================================================================== //
//                                 log::pos
// ============================================================================== //

/// `Position`: Mark source text position in file.
#[derive(Clone, Copy)]
pub struct Pos {
    line: u32,
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

/// Set default function for Pos.
impl Default for Pos {
    fn default() -> Self {
        Self::new()
    }
}

/// Set string for Pos.
impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

// ============================================================================== //
//                                 log::span
// ============================================================================== //

/// `Span`: Record source code position in file.
#[derive(Clone, Copy)]
pub struct Span {
    start: Pos,
    end: Pos,
}

impl Span {
    /// Tab space width: 4.
    const TAB_WIDTH: usize = 4;     

    /// Create a `Span` from `Pos`.
    pub fn new(pos : Pos) {
        Self { 
            start : pos, 
            end : pos
        };
    }

    /// Update end position.
    pub fn update_epos() {
        
    }

}


/// `get_line_str`: Get a line str and preprocess it.
/// 1. line -> str (`\t` -> `[space]`)
/// 2. line -> str (`\t` -> `[space]`, `col` update)
/// 3. line -> str (`\t` -> `[space]`, `col1` ~ `col2`)
macro_rules! get_line_str {
    ($line:expr) => {
        $line
            .map_or("".into(), |r| r.unwrap())
            .replace('\t', &format!("{:w$}", "", w = Span::TAB_WIDTH))
    };
    ($line:expr, $col:expr) => {
        let line = $line.map_or("".into(), |r| r.unwrap());
        let col = $col as usize;
        let tabs = (&line[..col]).matches('\t').count();
        (
            line.replace('\t', &format!("{:w$}", "", w = Span::TAB_WIDTH)),
            col + tabs * (Span::TAB_WIDTH - 1),
        )
    };
    ($line:expr, $col1:expr, $col2:expr) => {
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
    };
}
