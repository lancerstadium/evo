
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use std::{cell::RefCell, fmt::{Debug, Display}};


use crate::{ir::ty::{IRType, IRTypeKind}, log_warning};
use crate::util::log::Span;
use crate::{log_fatal, log_error};


// ============================================================================== //
//                              val::IRValue
// ============================================================================== //


/// `IRValue`: Value in the IR
/// - Support 8-bit, 16-bit, 32-bit, 64-bit and float, double
#[derive(Debug, Clone, PartialEq)]
pub struct IRValue {
    pub ty: IRType,
    pub val: RefCell<Vec<u8>>
}

impl IRValue {

    /// Create a new IRValue
    pub fn new(ty: IRType) -> IRValue {
        let size = ty.size();
        let buffer = vec![0; size];
        let val = RefCell::new(buffer);
        IRValue { ty, val }
    }

    // ==================== IRValue.ctl ==================== //

    /// Get the kind of the IRValue
    pub fn kind(&self) -> &IRTypeKind {
        self.ty.kind()
    }

    /// Get the size of the IRValue
    pub fn size(&self) -> usize {
        self.ty.size()
    }

    /// Get the scale of the IRValue
    pub fn scale(&self) -> Vec<usize> {
        self.ty.scale()
    }

    /// Get the sum of the scale of the IRValue
    pub fn scale_sum(&self) -> usize {
        self.ty.scale().iter().sum()
    }

    /// Get the bits hex string of the IRValue: Default little-endian
    /// - `index`: start index
    /// - `byte_num`: number of bytes you want, -1 means all
    pub fn hex(&self, index: usize, byte_num: i32) -> String {
        let mut num = byte_num as usize;
        // Check index and byte_num
        if byte_num < 0 {
            num = self.size() - index;
        }
        // Get index byte hexs
        let mut hexs = String::new();
        for i in 0..num {
            hexs.push_str(&format!("{:02X}", self.get_u8(index + i as usize)));
            if i < num - 1 {
                hexs.push(' ');
            }
        }
        hexs
    }

    /// Get the bits binary string of the IRValue: Default little-endian
    /// - `index`: start index
    /// - `byte_num`: number of bytes you want, -1 means all
    pub fn bin(&self, index: usize, byte_num: i32, big_endian: bool) -> String {
        let mut num = byte_num as usize;
        // Check index and byte_num
        if byte_num < 0 {
            num = self.size() - index;
        }
        // Get index byte bin
        let mut bin = String::new();
        for i in 0..num {
            if big_endian {
                bin.push_str(&format!("{:08b}", self.get_u8(index + num - 1 - i as usize)));
            } else {
                bin.push_str(&format!("{:08b}", self.get_u8(index + i as usize)));
            }
            if i < num - 1 {
                bin.push(' ');
            }
        }
        bin
    }
    /// Check IRValue size bound
    pub fn bound<T>(&self, index: usize, _: T) {
        // index + size of T should <= self.size()
        let is_valid = index + std::mem::size_of::<T>() <= self.size();
        if !is_valid {
            log_fatal!("Index out of bounds: index+type: {}+{} > size: {}", index, std::mem::size_of::<T>(), self.size());
        }
    }

    // ==================== IRValue.get ==================== //

    /// Get value by 8-bit
    pub fn get_u8(&self, index: usize) -> u8 {
        self.bound(index, 1 as u8);
        let buffer = self.val.borrow();
        u8::from_le_bytes([buffer[index]])
    }

    /// Get value by 16-bit
    pub fn get_u16(&self, index: usize) -> u16 {
        self.bound(index, 1 as u16);
        let buffer = self.val.borrow();
        u16::from_le_bytes([buffer[index], buffer[index + 1]])
    }

    /// Get value by 32-bit
    pub fn get_u32(&self, index: usize) -> u32 {
        self.bound(index, 1 as u32);
        let buffer = self.val.borrow();
        u32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by 64-bit
    pub fn get_u64(&self, index: usize) -> u64 {
        self.bound(index, 1 as u64);
        let buffer = self.val.borrow();
        u64::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7]])
    }

    /// Get value by signed 8-bit
    pub fn get_i8(&self, index: usize) -> i8 {
        self.bound(index, 1 as i8);
        let buffer = self.val.borrow();
        i8::from_le_bytes([buffer[index]])
    }

    /// Get value by signed 16-bit
    pub fn get_i16(&self, index: usize) -> i16 {
        self.bound(index, 1 as i16);
        let buffer = self.val.borrow();
        i16::from_le_bytes([buffer[index], buffer[index + 1]])
    }

    /// Get value by signed 32-bit
    pub fn get_i32(&self, index: usize) -> i32 {
        self.bound(index, 1 as i32);
        let buffer = self.val.borrow();
        i32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by signed 64-bit
    pub fn get_i64(&self, index: usize) -> i64 {
        self.bound(index, 1 as i64);
        let buffer = self.val.borrow();
        i64::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7]])
    }

    /// Get value by float
    pub fn get_f32(&self, index: usize) -> f32 {
        self.bound(index, 1 as f32);
        let buffer = self.val.borrow();
        f32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by double
    pub fn get_f64(&self, index: usize) -> f64 {
        self.bound(index, 1 as f64);
        let buffer = self.val.borrow();
        f64::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7]])
    }

    /// Get value by char
    pub fn get_char(&self, index: usize) -> char {
        self.get_u8(index) as char
    }

    /// Get value by string
    pub fn get_str(&self, index: usize) -> String {
        let buffer = self.val.borrow_mut();
        let mut str = String::new();
        for i in index..buffer.len() {
            if buffer[i] == 0 {
                break;
            }
            str.push(buffer[i] as char);
        }
        str
    }

    /// Get value by binary
    pub fn get_bin(&self, index: usize) -> Vec<u8> {
        let buffer = self.val.borrow();
        buffer[index..].to_vec()
    }


    // ==================== IRValue.set ==================== //

    /// Set type of the IRValue by `IRTypeKind`
    pub fn set_kind(&mut self, kind : IRTypeKind) {
        // 1. if kind equal to self.ty.kind, return
        if kind == *self.ty.kind() {
            return;
        }
        // 2. Change the type
        self.ty.set(kind);
        // 3. Expand the buffer
        let size = self.ty.size();
        let mut buffer = self.val.borrow_mut();
        while buffer.len() < size {
            buffer.push(0);
        }
    }

    /// Set type of the IRValue by `IRType`
    pub fn set_type(&mut self, ty : IRType) {
        // 1. Change the type
        self.ty = ty.clone();
        // 2. Expand the buffer
        let size = self.ty.size();
        let mut buffer = self.val.borrow_mut();
        while buffer.len() < size {
            buffer.push(0);
        }
    }

    /// Set value by bit
    /// `op_str`: `1`, `0`, `~`
    pub fn set_bit(&mut self, index: usize, start: usize, end: usize, op_str: &str) {
        if (start > end) || (end > self.size() * 8 - index * 8 - 1) {
            log_error!("Invalid start and end: {}, {}", start, end);
        }
        let mut buffer = self.val.borrow_mut();
        // Check how many u8 we need access
        let end_offset = end / 8 + 1;
        // Check how many index we can offset.
        let begin_offset = start / 8;
        // Get a vector like: start 3, end 9 -> [[3, 7], [0, 1]]
        // start 12, end 17 -> [[4, 7], [0, 1]]
        let mut idx_vec = Vec::new();
        if end_offset - begin_offset == 0 {
            log_error!("Invalid start and end: {}, {}", start, end);
            return
        } else if end_offset - begin_offset == 1 {
            idx_vec.push(vec![start % 8, end % 8]);
        } else if end_offset - begin_offset == 2 {
            idx_vec.push(vec![start % 8, 7]);
            idx_vec.push(vec![0, end % 8]);
        } else {
            idx_vec.push(vec![start % 8, 7]);
            for _ in 1..end_offset - 1 {
                idx_vec.push(vec![0, 7]);
            }
            idx_vec.push(vec![0, end % 8]);
        }
        assert!(idx_vec.len() == end_offset - begin_offset);
        // Set value
        match op_str {
            "1" => {
                for i in index + begin_offset..index + end_offset {
                    let mut value: u8 = 0;
                    let st = idx_vec[i - index - begin_offset][0];
                    let ed = idx_vec[i - index - begin_offset][1];
                    for j in st..=ed {
                        value |= 1 << j;
                    }
                    buffer[i] |= value;
                }
            },
            "0" => {
                for i in index + begin_offset..index + end_offset {
                    let mut value: u8 = 0;
                    let st = idx_vec[i - index - begin_offset][0];
                    let ed = idx_vec[i - index - begin_offset][1];
                    for j in st..=ed {
                        value |= 1 << j;
                    }
                    buffer[i] &= !value;
                }
            },
            "~" => {
                for i in index + begin_offset..index + end_offset {
                    let mut value: u8 = 0;
                    let st = idx_vec[i - index - begin_offset][0];
                    let ed = idx_vec[i - index - begin_offset][1];
                    for j in st..=ed {
                        value |= 1 << j;
                    }
                    buffer[i] ^= value;
                }
            },
            _ => {
                log_error!("Invalid bit op: {}", op_str);
            }
        }
    }

    /// Set value by byte: 8-bits
    /// `op_str`: `&`, `|`, `^`, `c`, `~c`, `shl`, `shr`, `rotl`, `rotr`
    pub fn set_byte(&mut self, index: usize, value: u8, op_str: &str) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        match op_str {
            "&" => {        // AND
                buffer[index] &= value;
            },
            "|" => {        // OR
                buffer[index] |= value;
            },
            "^" => {        // XOR
                buffer[index] ^= value;
            },
            "c" => {        // Cover
                buffer[index] = value;
            },
            "~c" => {       // Cover NOT
                buffer[index] = !value;
            },
            "shl" => {      // shift left
                buffer[index] <<= value % 8;
            },
            "shr" => {      // shift right
                buffer[index] >>= value % 8;
            },
            "rotl" => {     // rotate left
                buffer[index] = (buffer[index] << value % 8) | (buffer[index] >> (8 - value % 8));
            },
            "rotr" => {     // rotate right
                buffer[index] = (buffer[index] >> value % 8) | (buffer[index] << (8 - value % 8));
            },
            _ => {
                log_error!("Invalid byte op: {}", op_str);
            }
        }
    }
    
    /// Set value by bits string: `00001010`
    pub fn set_bits(&mut self, index: usize, value: &str, change_length: bool, big_endian: bool) {
        let mut val_size = self.size();
        // Set length
        if change_length {
            val_size = index + (value.len() as f64 / 8.0).ceil() as usize;
            self.set_kind(IRTypeKind::Array(IRType::u8(), val_size));
        }
        let mut cnt = 0;
        let mut idx;
        for i in 0..value.len() {
            // If index is out of range, break
            if i >= (val_size - index) * 8 {
                log_error!("Index out of range: {} >= {}", i, (val_size - index) * 8);
                break;
            }
            if big_endian {
                // Set bits: big-endian
                idx = val_size * 8 - 1 - cnt + (index * 8);
            } else {
                // Set bits: little-endian
                idx = cnt / 8 * 8 + (8 - cnt % 8) - 1 + (index * 8);
            }
            match value.chars().nth(i).unwrap() {
                '0' => {
                    self.set_bit(0, idx, idx, "0");
                    cnt += 1;
                },
                '1' => {
                    self.set_bit(0, idx, idx, "1");
                    cnt += 1;
                },
                '.' => {
                    cnt += 1;
                },
                // if got other char, jump
                _ => continue,
            }
        }
    }

    /// Set value by 8-bit
    pub fn set_8bit(&mut self, index: usize, value: u8) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
    }

    /// Set value by 16-bit
    pub fn set_16bit(&mut self, index: usize, value: u16) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
        buffer[index + 1] = bytes[1];
    }

    /// Set value by 32-bit
    pub fn set_32bit(&mut self, index: usize, value: u32) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
        buffer[index + 1] = bytes[1];
        buffer[index + 2] = bytes[2];
        buffer[index + 3] = bytes[3];
    }

    /// Set value by 64-bit
    pub fn set_64bit(&mut self, index: usize, value: u64) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
        buffer[index + 1] = bytes[1];
        buffer[index + 2] = bytes[2];
        buffer[index + 3] = bytes[3];
        buffer[index + 4] = bytes[4];
        buffer[index + 5] = bytes[5];
        buffer[index + 6] = bytes[6];
        buffer[index + 7] = bytes[7];
    }

    /// Set value by unsigned 8-bit
    pub fn set_u8(&mut self, index: usize, value: u8) {
        self.set_kind(IRTypeKind::U8);
        self.set_8bit(index, value);
    }

    /// Set value by unsigned 16-bit
    pub fn set_u16(&mut self, index: usize, value: u16) {
        self.set_kind(IRTypeKind::U16);
        self.set_16bit(index, value);
    }

    /// Set value by unsigned 32-bit
    pub fn set_u32(&mut self, index: usize, value: u32) {
        self.set_kind(IRTypeKind::U32);
        self.set_32bit(index, value);
    }

    /// Set value by unsigned 64-bit
    pub fn set_u64(&mut self, index: usize, value: u64) {
        self.set_kind(IRTypeKind::U64);
        self.set_64bit(index, value);
    }

    /// Set value by signed 8-bit
    pub fn set_i8(&mut self, index: usize, value: i8) {
        self.set_kind(IRTypeKind::I8);
        self.set_8bit(index, value as u8);
    }

    /// Set value by signed 16-bit
    pub fn set_i16(&mut self, index: usize, value: i16) {
        self.set_kind(IRTypeKind::I16);
        self.set_16bit(index, value as u16);
    }

    /// Set value by signed 32-bit
    pub fn set_i32(&mut self, index: usize, value: i32) {
        self.set_kind(IRTypeKind::I32);
        self.set_32bit(index, value as u32);
    }

    /// Set value by signed 64-bit
    pub fn set_i64(&mut self, index: usize, value: i64) {
        self.set_kind(IRTypeKind::I64);
        self.set_64bit(index, value as u64);
    }

    /// Set value by float
    pub fn set_f32(&mut self, index: usize, value: f32) {
        self.set_kind(IRTypeKind::F32);
        let bytes = value.to_le_bytes();
        self.set_8bit(index, bytes[0]);
        self.set_8bit(index + 1, bytes[1]);
        self.set_8bit(index + 2, bytes[2]);
        self.set_8bit(index + 3, bytes[3]);
    }

    /// Set value by double
    pub fn set_f64(&mut self, index: usize, value: f64) {
        self.set_kind(IRTypeKind::F64);
        let bytes = value.to_le_bytes();
        self.set_8bit(index, bytes[0]);
        self.set_8bit(index + 1, bytes[1]);
        self.set_8bit(index + 2, bytes[2]);
        self.set_8bit(index + 3, bytes[3]);
        self.set_8bit(index + 4, bytes[4]);
        self.set_8bit(index + 5, bytes[5]);
        self.set_8bit(index + 6, bytes[6]);
        self.set_8bit(index + 7, bytes[7]);
    }

    /// Set value by array
    pub fn set_array(&mut self, value: Vec<IRValue>) {
        self.set_kind(IRTypeKind::Array(value[0].ty.clone(), value.len()));
        let size = self.size();
        assert_eq!(value.len() * value[0].size(), size);
        let mut buffer = self.val.borrow_mut();
        // value: [1,2], [3,0], [2,4] -> buffer: [1,2,3,0,2,4]
        for i in 0..value.len() {
            let val_bytes = value[i].val.borrow();
            for j in 0..value[i].size() {
                buffer[i * value[i].size() + j] = val_bytes[j];
            }
        }
    }

    /// Set value by tuple
    pub fn set_tuple(&mut self, value: Vec<IRValue>) {
        // Get Vec<IRType> from Vec<IRValue>
        let mut types = Vec::new();
        let mut types_size = 0;
        for val in value.iter() {
            types.push(val.ty.clone());
            types_size += val.size();
        }
        self.set_kind(IRTypeKind::Tuple(types));
        let size = self.size();
        assert_eq!(types_size, size);
        // (1,3,4), (2), (3,4) -> buffer: [1,3,4,2,3,4]
        let mut buffer = self.val.borrow_mut();
        let mut type_size = 0;
        for i in 0..value.len() {
            let val_bytes = value[i].val.borrow();
            for j in 0..value[i].size() {
                buffer[type_size + j] = val_bytes[j];
            }
            type_size += value[i].size();
        }
    }

    /// Set value by pointer/struct/tuple
    pub fn set_ptr(&mut self, value: IRValue) {
        self.set_kind(IRTypeKind::Ptr(value.ty.clone()));
        let size = self.size();
        assert_eq!(value.size(), size);
        let mut buffer = self.val.borrow_mut();
        for i in 0..size {
            buffer[i] = value.val.borrow()[i];
        }
    }

    /// Set value by char(4-bits)
    pub fn set_char(&mut self, index: usize, value: char) {
        self.set_8bit(index, value as u8);
    }

    /// Set value by string
    pub fn set_str(&mut self, value: &str) {
        self.set_kind(IRTypeKind::Array(IRType::u8(), value.len()));
        let size = self.size();
        assert_eq!(value.len(), size);
        let mut buffer = self.val.borrow_mut();
        for i in 0..value.len() {
            buffer[i] = value.as_bytes()[i];
        }
    }

    /// Set zero.
    pub fn set_zero(&mut self) {
        let size = self.size();
        let mut buffer = self.val.borrow_mut();
        for i in 0..size {
            buffer[i] = 0;
        }
    }

    // ==================== IRValue.is ===================== //

    /// Check str if the value is u8 number
    pub fn is_u8(str : &str) -> bool {
        str.parse::<u8>().is_ok()
    }

    /// Check str if the value is u16 number
    pub fn is_u16(str : &str) -> bool {
        str.parse::<u16>().is_ok()
    }

    /// Check str if the value is u32 number
    pub fn is_u32(str : &str) -> bool {
        str.parse::<u32>().is_ok()
    }

    /// Check str if the value is u64 number
    pub fn is_u64(str : &str) -> bool {
        str.parse::<u64>().is_ok()
    }

    /// Check str if the value is i8 number
    pub fn is_i8(str : &str) -> bool {
        str.parse::<i8>().is_ok()
    }

    /// Check str if the value is i16 number
    pub fn is_i16(str : &str) -> bool {
        str.parse::<i16>().is_ok()
    }

    /// Check str if the value is i32 number
    pub fn is_i32(str : &str) -> bool {
        str.parse::<i32>().is_ok()
    }

    /// Check str if the value is i64 number
    pub fn is_i64(str : &str) -> bool {
        str.parse::<i64>().is_ok()
    }

    /// Check str if the value is float number
    pub fn is_f32(str : &str) -> bool {
        str.parse::<f32>().is_ok()
    }

    /// Check str if the value is double number
    pub fn is_f64(str : &str) -> bool {
        str.parse::<f64>().is_ok()
    }

    /// Check str if the value is array
    pub fn is_array(str : &str) -> bool {
        // begin with `[` and end with `]`
        str.starts_with('[') && str.ends_with(']')
    }

    /// Check str if the value is string
    pub fn is_str(str : &str) -> bool {
        // begin with `"` and end with `"`
        str.starts_with('"') && str.ends_with('"')
    }

    /// Check str if the value is tuple
    pub fn is_tuple(str : &str) -> bool {
        // begin with `(` and end with `)` and has `,`
        str.starts_with('(') && str.ends_with(')') && str.contains(',')
    }

    /// Check str if the value is pointer: tuple / struct / function
    pub fn is_ptr(str : &str) -> bool {
        Self::is_struct(str) || Self::is_func(str)
    }

    /// Check str if the value is function
    pub fn is_func(str : &str) -> bool {
        // begin with `(` and has `) ->`
        str.starts_with('(') && str.contains(") ->")
    }

    /// Check str if the value is struct
    pub fn is_struct(str : &str) -> bool {
        // begin with `struct {` and end with `}` and has `,`
        str.starts_with("struct {") && str.ends_with('}') && str.contains(',')
    }

    /// Check str if the value is hex
    pub fn is_hex(str : &str) -> bool {
        // begin with `0x`
        str.starts_with("0x")
    }

    /// Check str if the value is bin
    pub fn is_bin(str : &str) -> bool {
        // begin with `0b` or `0B`
        str.starts_with("0b") || str.starts_with("0B")
    }
    

    // ==================== IRValue.from =================== //


    /// Get bits from String: `0b00001010 00001010`
    pub fn bits(value: &str) -> IRValue {
        let is_big_endian = value.starts_with("0B");

        let value = value[2..].to_string();
        let value = value.trim();
        // Deal with value string: `0b00001010` -> `10`
        // Get value length
        let len = value.len();
        // Parse value string: `0` -> 0, `1` -> 1, `.` -> ., other jump
        let mut new_val = String::new();
        for i in 0..len {
            if value.chars().nth(i).unwrap() == '0' {
                new_val += "0";
            } else if value.chars().nth(i).unwrap() == '1' {
                new_val += "1";
            } else if value.chars().nth(i).unwrap() == '.' {
                new_val += ".";
            } else {
                continue;
            }
        }
        // Get val size: Upper bound
        let val_size = (new_val.len() as f64 / 8.0).ceil() as usize;
        let mut val = IRValue::new(IRType::array(IRType::u8(), val_size));

        // Set value
        val.set_bits(0, &new_val, true, is_big_endian);
        val
    }

    /// Get value from u8
    pub fn u8(value: u8) -> IRValue {
        let mut val = IRValue::new(IRType::u8());
        val.set_u8(0, value);
        val
    }

    /// Get value from u16
    pub fn u16(value: u16) -> IRValue {
        let mut val = IRValue::new(IRType::u16());
        val.set_u16(0, value);
        val
    }

    /// Get value from u32
    pub fn u32(value: u32) -> IRValue {
        let mut val = IRValue::new(IRType::u32());
        val.set_u32(0, value);
        val
    }

    /// Get value from u64
    pub fn u64(value: u64) -> IRValue {
        let mut val = IRValue::new(IRType::u64());
        val.set_u64(0, value);
        val
    }

    /// Get value from i8
    pub fn i8(value: i8) -> IRValue {
        let mut val = IRValue::new(IRType::i8());
        val.set_i8(0, value);
        val
    }

    /// Get value from i16
    pub fn i16(value: i16) -> IRValue {
        let mut val = IRValue::new(IRType::i16());
        val.set_i16(0, value);
        val
    }

    /// Get value from i32
    pub fn i32(value: i32) -> IRValue {
        let mut val = IRValue::new(IRType::i32());
        val.set_i32(0, value);
        val
    }

    /// Get value from i64
    pub fn i64(value: i64) -> IRValue {
        let mut val = IRValue::new(IRType::i64());
        val.set_i64(0, value);
        val
    }

    /// Get value from float
    pub fn f32(value: f32) -> IRValue {
        let mut val = IRValue::new(IRType::f32());
        val.set_f32(0, value);
        val
    }

    /// Get value from double
    pub fn f64(value: f64) -> IRValue {
        let mut val = IRValue::new(IRType::f64());
        val.set_f64(0, value);
        val
    }

    /// Get value from array
    pub fn array(value: Vec<IRValue>) -> IRValue {
        // Set kind
        let mut val = IRValue::new(IRType::array(value[0].ty.clone(), value.len()));
        val.set_array(value);
        val
    }

    /// Get value from tuple
    pub fn tuple(value: Vec<IRValue>) -> IRValue {
        // Get Vec<IRType> from Vec<IRValue>
        let mut types = Vec::new();
        for val in value.iter() {
            types.push(val.ty.clone());
        }
        // Set kind
        let mut val = IRValue::new(IRType::tuple(types));
        val.set_tuple(value);
        val
    }

    /// Get value from pointer
    pub fn ptr(value: IRValue) -> IRValue {
        let mut val = IRValue::new(IRType::ptr(value.ty.clone()));
        val.set_ptr(value);
        val
    }

    /// Get value from text string -> array: `"hello"`
    pub fn str(value: &str) -> IRValue {
        let mut val = IRValue::new(IRType::array(IRType::u8(), value.len()));
        val.set_str(value);
        val
    }

    /// Get value from string
    pub fn from_string(value: &str) -> IRValue {
        let value = value.trim();
        if IRValue::is_i32(value) { // parse as i32
            return IRValue::i32(value.parse::<i32>().unwrap());
        } else if IRValue::is_i64(value) { // parse as i64
            return IRValue::i64(value.parse::<i64>().unwrap());
        } else if IRValue::is_f32(value) { // parse as f32
            return IRValue::f32(value.parse::<f32>().unwrap());
        } else if IRValue::is_f64(value) { // parse as f64
            return IRValue::f64(value.parse::<f64>().unwrap());
        } else if IRValue::is_array(value) { // parse as array
            // Deal with value string: `[1, 2, 3]`
            let value = value[1..value.len() - 1].to_string();
            let value = value.trim();
            let value = value.split(',').map(|v| v.trim().to_string()).collect::<Vec<String>>();
            let value = value.iter().map(|v| IRValue::from_string(v)).collect::<Vec<IRValue>>();
            return IRValue::array(value);
        } else if IRValue::is_tuple(value) { // parse as tuple
            // Deal with value string: `(1, 2.7, 3)`
            let value = value[1..value.len() - 1].to_string();
            let value = value.trim();
            let value = value.split(',').map(|v| v.trim().to_string()).collect::<Vec<String>>();
            let value = value.iter().map(|v| IRValue::from_string(v)).collect::<Vec<IRValue>>();
            return IRValue::tuple(value);
        } else if IRValue::is_str(value) { // parse as string
            // Deal with value string: `"hello"`
            // Delete `"`, `"` and delete `\n` `\r` `\t` on the side
            let value = value[1..value.len() - 1].to_string();
            let value = value.trim();
            return IRValue::str(&value);
        } else if IRValue::is_hex(value) { // parse as hex
            let value = value[2..].to_string();
            let value = value.trim();
            return IRValue::u8(value.parse::<u8>().unwrap());
        } else if IRValue::is_bin(value) { // parse as bin
            return IRValue::bits(value);
        } else {
            log_warning!("Can't parse {} as IRValue", value);
            IRValue::i32(0)
        }
    }

}

impl Default for IRValue {
    /// Default value type is i32
    fn default() -> Self {
        IRValue::i32(0)
    }
}

impl Display for IRValue {
    /// Display the IRValue by IRType
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            IRTypeKind::U8 => write!(f, "{}", self.get_u8(0)),
            IRTypeKind::U16 => write!(f, "{}", self.get_u16(0)),
            IRTypeKind::U32 => write!(f, "{}", self.get_u32(0)),
            IRTypeKind::U64 => write!(f, "{}", self.get_u64(0)),
            IRTypeKind::I8 => write!(f, "{}", self.get_i8(0)),
            IRTypeKind::I16 => write!(f, "{}", self.get_i16(0)),
            IRTypeKind::I32 => write!(f, "{}", self.get_i32(0)),
            IRTypeKind::I64 => write!(f, "{}", self.get_i64(0)),
            IRTypeKind::F32 => write!(f, "{}", self.get_f32(0)),
            IRTypeKind::F64 => write!(f, "{}", self.get_f64(0)),
            _ => write!(f, "{}", self.hex(0, -1)),
        }
    }
}

impl Eq for IRValue {}



// ============================================================================== //
//                               Unit Tests
// ============================================================================== //


#[cfg(test)]
mod val_tests {

    use super::*;

    #[test]
    fn val_get() {
        let val = IRValue::u8(12 as u8);
        assert_eq!(val.get_u8(0), 12 as u8);
        let val = IRValue::i8(-12 as i8);
        assert_eq!(val.get_i8(0), -12 as i8);
        let val = IRValue::u16(12 as u16);
        assert_eq!(val.get_u16(0), 12 as u16);
        let val = IRValue::i16(-12 as i16);
        assert_eq!(val.get_i16(0), -12 as i16);
        let val = IRValue::u32(12 as u32);
        assert_eq!(val.get_u32(0), 12 as u32);
        let val = IRValue::i32(-12 as i32);
        assert_eq!(val.get_i32(0), -12 as i32);
        let val = IRValue::u64(129 as u64);
        assert_eq!(val.get_u64(0), 129 as u64);
        let val = IRValue::i64(-12 as i64);
        assert_eq!(val.get_i64(0), -12 as i64);
        let val = IRValue::f32(12.33f32);
        assert_eq!(val.get_f32(0), 12.33f32);
        let val = IRValue::f64(12.34f64);
        assert_eq!(val.get_f64(0), 12.34f64);
    }

    #[test]
    fn val_print() {
        let mut val = IRValue::new(IRType::u8());
        val.set_u8(0, 9 as u8);
        assert_eq!(val.hex(0, -1), "09");

        // Change type
        val.set_i64(0, 255 as i64);
        assert_eq!(val.hex(0, -1), "FF 00 00 00 00 00 00 00");
        assert_eq!(val.kind().to_string(), IRTypeKind::I64.to_string());
        assert_eq!(val.ty, IRType::i64());

        // Only Write in data, don't change type
        val.set_8bit(0, 64 as u8);
        assert_eq!(val.hex(0, -1), "40 00 00 00 00 00 00 00");
        assert_eq!(val.to_string(), "64");

        // Check binary
        let mut val = IRValue::from_string("1296");
        assert_eq!(val.get_i32(0), 1296 as i32);
        assert_eq!(val.bin(0, -1, false), "00010000 00000101 00000000 00000000");
        val.set_bit(0, 3, 18, "1");
        assert_eq!(val.bin(0, -1, false), "11111000 11111111 00000111 00000000");
        val.set_bit(1, 0, 7, "~");
        assert_eq!(val.bin(0, -1, false), "11111000 00000000 00000111 00000000");
        val.set_byte(2, 9, "^");    // need to enhance this
        assert_eq!(val.bin(0, -1, false), "11111000 00000000 00001110 00000000");

        // `set_type` by array
        val.set_type(IRType::array(IRType::u32(), 3));
        assert_eq!(val.kind().to_string(), "[u32; 3]");
        assert_eq!(val.size(), 12);

        // `set_type` by tuple
        val.set_type(IRType::tuple(vec![IRType::u32(), IRType::u64()]));
        assert_eq!(val.kind().to_string(), "(u32, u64)");
        assert_eq!(val.size(), 12);

    }

    #[test]
    fn val_from() {

        let val = IRValue::from_string("23.5");
        assert_eq!(val.get_f32(0), 23.5f32);

        let val = IRValue::from_string("[18, 22, 623]");
        assert_eq!(val.get_i32(0), 18);
        assert_eq!(val.get_i32(4), 22);
        assert_eq!(val.get_i32(8), 623);

        let val = IRValue::from_string("[-18, 1, -23]");
        assert_eq!(val.get_i32(0), -18);
        assert_eq!(val.get_i32(4), 1);
        assert_eq!(val.get_i32(8), -23);

        let val = IRValue::from_string("[-18.5, 1.5, -23.5]");
        assert_eq!(val.get_f32(0), -18.5f32);
        assert_eq!(val.get_f32(4), 1.5f32);
        assert_eq!(val.get_f32(8), -23.5f32);

        let val = IRValue::from_string("(-18.5, 0, -23.5, \"hello\")");
        assert_eq!(val.get_f32(0), -18.5f32);
        assert_eq!(val.get_i32(4), 0);
        assert_eq!(val.get_f32(8), -23.5f32);
        assert_eq!(val.get_str(13), "ello");

        let mut val = IRValue::from_string("\"hallo\"");
        val.set_char(1, 'e');
        assert_eq!(val.get_char(1), 'e');
        assert_eq!(val.get_str(1), "ello");

        let val = IRValue::from_string("0b00001010 11001010 1111");
        assert_eq!(val.bin(0, -1, false), "00001010 11001010 11110000");
        let mut val1 = IRValue::from_string("0b00010000 00000101 00000000 00000000");
        let val2 = IRValue::from_string("0B00010000 00000101 00000000 00000000");
        assert_eq!(val1.bin(0, -1, true), val2.bin(0, -1, false));
        assert_eq!(val1.bin(0, -1, false), val2.bin(0, -1, true));

        assert_eq!(val1.bin(0, -1, false), "00010000 00000101 00000000 00000000");
        val1.set_bits(1, "10..1.10 ...111.0", false, false);
        assert_eq!(val1.bin(0, -1, false), "00010000 10001110 00011100 00000000");
    }


    #[test]
    fn val_change() {
        let mut val = IRValue::new(IRType::u8());
        val.set_u8(0, 1 as u8);
        assert_eq!(val.get_u8(0), 1 as u8);
        // Change value
        val.set_u8(0, 2 as u8);
        assert_eq!(val.get_u8(0), 2 as u8);
        // Change Type
        val.set_i8(0, -3 as i8);
        assert_eq!(val.get_i8(0), -3 as i8);
        // Change to Float
        val.set_f32(0, 255.33f32);
        assert_eq!(val.get_f32(0), 255.33f32);
        // Change to Double
        val.set_f64(0, 255.34f64);
        assert_eq!(val.get_f64(0), 255.34f64);
    }

}