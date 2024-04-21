
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use std::{cell::RefCell, fmt::{Debug, Display}};


use crate::{ir::ty::{IRType, IRTypeKind}, log_warning};
use crate::util::log::Span;
use crate::log_fatal;

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
        assert!(self.ty.size() == self.val.borrow().len());
        self.ty.size()
    }

    /// Get the bits hex string of the IRValue: Default little-endian
    pub fn hex(&self) -> String {
        self.val.borrow().iter().map(|x| format!("{:02X}", x)).collect::<Vec<String>>().join(" ")
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

    /// Check str if the value is pointer: tuple / struct / function
    pub fn is_ptr(str : &str) -> bool {
        Self::is_tuple(str) || Self::is_struct(str) || Self::is_func(str)
    }

    /// Check str if the value is tuple
    pub fn is_tuple(str : &str) -> bool {
        // begin with `(` and end with `)` and has `,`
        str.starts_with('(') && str.ends_with(')') && str.contains(',')
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
    

    // ==================== IRValue.from =================== //

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

    /// Get value from pointer
    pub fn ptr(value: IRValue) -> IRValue {
        let mut val = IRValue::new(IRType::ptr(value.ty.clone()));
        val.set_ptr(value);
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
        } else {
            log_warning!("Can't parse {} as IRValue", value);
        }
        IRValue::i32(0)
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
            _ => write!(f, "{}", self.hex()),
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
    fn val_print() {
        let mut val = IRValue::new(IRType::u8());
        val.set_u8(0, 9 as u8);
        assert_eq!(val.hex(), "09");

        // Change type
        val.set_i64(0, 255 as i64);
        assert_eq!(val.hex(), "FF 00 00 00 00 00 00 00");
        assert_eq!(val.kind().to_string(), IRTypeKind::I64.to_string());
        assert_eq!(val.ty, IRType::i64());

        // Only Write in data, don't change type
        val.set_8bit(0, 64 as u8);
        assert_eq!(val.hex(), "40 00 00 00 00 00 00 00");
        assert_eq!(val.to_string(), "64");

        // `set_type` by array
        val.set_type(IRType::array(IRType::u32(), 3));
        assert_eq!(val.kind().to_string(), "[u32; 3]");
        assert_eq!(val.size(), 12);

    }

    #[test]
    fn val_from() {
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


        let val = IRValue::from_string("12");
        assert_eq!(val.get_i32(0), 12 as i32);

        let val = IRValue::from_string("23.5");
        assert_eq!(val.get_f32(0), 23.5f32);

        let val = IRValue::from_string("[18, 22, 623]");
        assert_eq!(val.get_i32(0), 18);
        assert_eq!(val.get_i32(4), 22);
        assert_eq!(val.get_i32(8), 623);

        let val = IRValue::from_string("[-18, 1, -23]");
        assert_eq!(val.get_i32(0), -18);
        assert_eq!(val.get_u32(4), 1);
        assert_eq!(val.get_i32(8), -23);
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