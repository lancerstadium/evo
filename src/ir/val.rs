
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use std::{cell::RefCell, fmt::{Debug, Display}};

use crate::ir::ty::{IRType, IRTypeKind};

use crate::util::log::Span;
use crate::log_fatal;

// ============================================================================== //
//                              val::IRValue
// ============================================================================== //


/// `IRValue`: Value in the IR
/// - Support 8-bit, 16-bit, 32-bit, 64-bit and float, double
#[derive(Debug, Clone)]
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
        self.val.borrow()[index]
    }

    /// Get value by 16-bit
    pub fn get_u16(&self, index: usize) -> u16 {
        self.bound(index, 1 as u16);
        let buffer = self.val.borrow();
        (buffer[index] as u16) | ((buffer[index + 1] as u16) << 8)
    }

    /// Get value by 32-bit
    pub fn get_u32(&self, index: usize) -> u32 {
        self.bound(index, 1 as u32);
        let buffer = self.val.borrow();
        (buffer[index] as u32) | ((buffer[index + 1] as u32) << 8) |
            ((buffer[index + 2] as u32) << 16) | ((buffer[index + 3] as u32) << 24)
    }

    /// Get value by 64-bit
    pub fn get_u64(&self, index: usize) -> u64 {
        self.bound(index, 1 as u64);
        let buffer = self.val.borrow();
        (buffer[index] as u64) | ((buffer[index + 1] as u64) << 8) |
            ((buffer[index + 2] as u64) << 16) | ((buffer[index + 3] as u64) << 24) |
            ((buffer[index + 4] as u64) << 32) | ((buffer[index + 5] as u64) << 40) |
            ((buffer[index + 6] as u64) << 48) | ((buffer[index + 7] as u64) << 56)
    }

    /// Get value by signed 8-bit
    pub fn get_i8(&self, index: usize) -> i8 {
        self.get_u8(index) as i8
    }

    /// Get value by signed 16-bit
    pub fn get_i16(&self, index: usize) -> i16 {
        self.get_u16(index) as i16
    }

    /// Get value by signed 32-bit
    pub fn get_i32(&self, index: usize) -> i32 {
        self.get_u32(index) as i32
    }

    /// Get value by signed 64-bit
    pub fn get_i64(&self, index: usize) -> i64 {
        self.get_u64(index) as i64
    }

    /// Get value by float
    pub fn get_f32(&self, index: usize) -> f32 {
        self.get_u32(index) as f32
    }

    /// Get value by double
    pub fn get_f64(&self, index: usize) -> f64 {
        self.get_u64(index) as f64
    }


    // ==================== IRValue.set ==================== //

    /// Set type of the IRValue by `IRTypeKind`
    pub fn set_kind(&mut self, kind : IRTypeKind) {
        // 1. Change the type
        self.ty.set(kind);
        // 2. Expand the buffer
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
        buffer[index] = value;
    }

    /// Set value by 16-bit
    pub fn set_16bit(&mut self, index: usize, value: u16) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        buffer[index] = (value & 0xFF) as u8;
        buffer[index + 1] = ((value >> 8) & 0xFF) as u8;
    }

    /// Set value by 32-bit
    pub fn set_32bit(&mut self, index: usize, value: u32) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        buffer[index] = (value & 0xFF) as u8;
        buffer[index + 1] = ((value >> 8) & 0xFF) as u8;
        buffer[index + 2] = ((value >> 16) & 0xFF) as u8;
        buffer[index + 3] = ((value >> 24) & 0xFF) as u8;
    }

    /// Set value by 64-bit
    pub fn set_64bit(&mut self, index: usize, value: u64) {
        self.bound(index, value);
        let mut buffer = self.val.borrow_mut();
        buffer[index] = (value & 0xFF) as u8;
        buffer[index + 1] = ((value >> 8) & 0xFF) as u8;
        buffer[index + 2] = ((value >> 16) & 0xFF) as u8;
        buffer[index + 3] = ((value >> 24) & 0xFF) as u8;
        buffer[index + 4] = ((value >> 32) & 0xFF) as u8;
        buffer[index + 5] = ((value >> 40) & 0xFF) as u8;
        buffer[index + 6] = ((value >> 48) & 0xFF) as u8;
        buffer[index + 7] = ((value >> 56) & 0xFF) as u8;
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
        self.set_32bit(index, value as u32);
    }

    /// Set value by double
    pub fn set_f64(&mut self, index: usize, value: f64) {
        self.set_kind(IRTypeKind::F64);
        self.set_64bit(index, value as u64);
    }

    

}

impl Default for IRValue {
    /// Default value type is i32
    fn default() -> Self {
        IRValue::new(IRType::i32())
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
        val.set_f32(0, 255.0f32);
        assert_eq!(val.get_f32(0), 255.0f32);

        // Change to Double
        val.set_f64(0, 255.0f64);
        assert_eq!(val.get_f64(0), 255.0f64);
    }

}