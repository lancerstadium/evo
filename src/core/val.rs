
// ============================================================================== //
//                                 Use Mods
// ============================================================================== //


use std::{cell::RefCell, fmt::{Debug, Display}, rc::Rc};


use crate::{core::ty::{Types, TypesKind}, log_warning};
use crate::util::log::Span;
use crate::{log_fatal, log_error};


// ============================================================================== //
//                              val::Value
// ============================================================================== //


/// `Value`: Value in the IR
/// - Support 8-bit, 16-bit, 32-bit, 64-bit and float, double
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: Option<String>,
    pub ty: Types,
    pub val: RefCell<Vec<u8>>,
    pub ref_val: Option<Rc<RefCell<Value>>>,
    /// ### Value flag:
    /// ```txt
    /// 0 0 0 0 0 0 0 0
    /// │ │ │ │ │ │ │ │
    /// │ │ │ │ │ │ │ └── is constant
    /// │ │ │ │ │ │ └──── is mutable
    /// │ │ │ │ │ └────── is pointer
    /// │ │ │ │ └──────── <reserved>
    /// │ │ │ └────────── 0: is align by u8, 1: is strict align
    /// │ │ └──────────── <reserved>
    /// │ └────────────── <reserved>
    /// └──────────────── <reserved>
    /// ```
    pub flag: u16
}

impl Value {

    /// Create a new Value
    pub fn new(ty: Types) -> Value {
        let size = ty.size();
        let buffer = vec![0; size];
        let val = RefCell::new(buffer);
        Value { name: None, ty, val, ref_val: None, flag:0 }
    }

    /// Fill value by types, val_str and create a new Value
    pub fn fill_string(val_str: &str, ty: Option<Types>) -> Value {
        // check type
        let ty = if ty.is_none() { Types::none() } else { ty.unwrap() };
        // match types and parse val_str
        let val_str = val_str.trim();
        match ty.kind() {
            TypesKind::None => {
                if Value::is_i32(val_str) {
                    Value::fill_string(val_str, Some(Types::i32()))
                } else if Value::is_i64(val_str) {
                    Value::fill_string(val_str, Some(Types::i64()))
                } else if Value::is_f32(val_str) {
                    Value::fill_string(val_str, Some(Types::f32()))
                } else if Value::is_f64(val_str) {
                    Value::fill_string(val_str, Some(Types::f64()))
                } else if Value::is_array(val_str) {
                    // Deal with value string: `[1, 2, 3]`
                    let val_str = val_str[1..val_str.len() - 1].to_string();
                    let val_str = val_str.trim();
                    let val_str = val_str.split(',').map(|v| v.trim().to_string()).collect::<Vec<String>>();
                    let value = val_str.iter().map(|v| Value::fill_string(v, Some(ty.clone()))).collect::<Vec<Value>>();
                    // check value elem size
                    if value.iter().any(|v| v.ty.size() != ty.size()) {
                        log_warning!("Value elem size not match: {}", val_str.iter().map(|v| v.as_str()).collect::<Vec<&str>>().join(", "));
                        Value::tuple(value)
                    } else {
                        Value::array(value)
                    }
                } else if Value::is_tuple(val_str) {
                    // Deal with value string: `(1, 2, 3)`
                    let val_str = val_str[1..val_str.len() - 1].to_string();
                    let val_str = val_str.trim();
                    let val_str = val_str.split(',').map(|v| v.trim().to_string()).collect::<Vec<String>>();
                    let value: Vec<Value> = val_str.iter().map(|v| Value::fill_string(v, Some(ty.clone()))).collect::<Vec<Value>>();
                    Value::tuple(value)
                } else if Value::is_str(val_str) {
                    // Deal with value string: `"hello"`
                    // Delete `"`, `"` and delete `\n` `\r` `\t` on the side
                    let val_str = val_str[1..val_str.len() - 1].trim();
                    Value::str(val_str)
                } else if Value::is_bin(val_str) {
                    Value::bits(val_str)
                } else if Value::is_hex(val_str) {
                    Value::hexs(val_str)
                } else if Value::is_ptr(val_str) {
                    Value::ptr(Some(Value::fill_string(val_str, Some(ty.clone()))), None)
                } else if Value::is_ptr_ref(val_str){
                    // del `&`
                    let val_str = val_str[1..val_str.len()].trim();
                    Value::ptr(None, Some(Value::fill_string(val_str, Some(ty.clone()))))
                } else if val_str.is_empty() {
                    Value::bits(val_str)
                } else {
                    log_error!("Invalid value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::Void => {
                Value::new(ty)
            },
            TypesKind::I8 => {
                if Value::is_i8(val_str) {
                    Value::i8(val_str.parse::<i8>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::i8(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid i8 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::I16 => {
                if Value::is_i16(val_str) {
                    Value::i16(val_str.parse::<i16>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::i16(0);
                    val.set_name(val_str.to_string());
                    val
                }  else {
                    log_error!("Invalid i16 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::I32 => {
                if Value::is_i32(val_str) {
                    Value::i32(val_str.parse::<i32>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::i32(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid i32 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::I64 => {
                if Value::is_i64(val_str) {
                    Value::i64(val_str.parse::<i64>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::i64(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid i64 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::I128 => {
                if Value::is_i128(val_str) {
                    Value::i128(val_str.parse::<i128>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::i128(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid i128 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::U8 => {
                if Value::is_u8(val_str) {
                    Value::u8(val_str.parse::<u8>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::u8(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid u8 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::U16 => {
                if Value::is_u16(val_str) {
                    Value::u16(val_str.parse::<u16>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::u16(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid u16 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::U32 => {
                if Value::is_u32(val_str) {
                    Value::u32(val_str.parse::<u32>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::u32(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid u32 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::U64 => {
                if Value::is_u64(val_str) {
                    Value::u64(val_str.parse::<u64>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::u64(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid u64 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::U128 => {
                if Value::is_u128(val_str) {
                    Value::u128(val_str.parse::<u128>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::u128(0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid u128 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::F32 => {
                if Value::is_f32(val_str) {
                    Value::f32(val_str.parse::<f32>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::f32(0.0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid f32 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::F64 => {
                if Value::is_f64(val_str) {
                    Value::f64(val_str.parse::<f64>().unwrap())
                } else if Value::is_ident(val_str) {
                    let mut val = Value::f64(0.0);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid f64 value: {}", val_str);
                    Value::new(ty)
                }
            },
            TypesKind::Bit(width) => {
                if Value::is_bin(val_str) {
                    Value::bits(val_str)
                } else if Value::is_hex(val_str) {
                    Value::hexs(val_str)
                } else if Value::is_i32(val_str) {
                    let mut val = Value::i32(val_str.parse::<i32>().unwrap());
                    val.resize(*width);
                    val
                } else if Value::is_i64(val_str) {
                    let mut val = Value::i64(val_str.parse::<i64>().unwrap());
                    val.resize(*width);
                    val
                } else if Value::is_ident(val_str) {
                    let val = Value::bit(*width, 0);
                    val
                } else {
                    log_error!("Invalid bit value: {}", val_str);
                    Value::new(ty.clone())
                }
            },
            TypesKind::Array(ty, len) => {
                if Value::is_array(val_str) {
                    // Deal with value string: `[1, 2, 3]`
                    let val_str = val_str[1..val_str.len() - 1].to_string();
                    let val_str = val_str.trim();
                    let val_str = val_str.split(',').map(|v| v.trim().to_string()).collect::<Vec<String>>();
                    let mut value = val_str.iter().map(|v| Value::fill_string(v, Some(ty.clone()))).collect::<Vec<Value>>();
                    // fill len with value 0
                    if value.len() < *len {
                        for _ in value.len()..*len {
                            value.push(Value::new(ty.clone()));
                        }
                    }
                    Value::array(value)
                } else if Value::is_ident(val_str) {
                    let mut val = Value::array(vec![Value::new(ty.clone()); *len]);
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid array value: {}", val_str);
                    Value::new(ty.clone())
                }
            },
            TypesKind::Tuple(tys) => {
                if Value::is_tuple(val_str) {
                    // Deal with value string: `(1, 2, 3)`
                    let val_str = val_str[1..val_str.len() - 1].to_string();
                    let val_str = val_str.trim();
                    let val_str = val_str.split(',').map(|v| v.trim().to_string()).collect::<Vec<String>>();
                    let mut value: Vec<Value> = Vec::new();
                    for (i, ty) in tys.iter().enumerate() {
                        // every element match type
                        let value_elem = Value::fill_string(val_str[i].as_str(), Some(ty.clone()));
                        value.push(value_elem);
                    }
                    Value::tuple(value)
                } else if Value::is_ident(val_str) {
                    let mut val = Value::tuple(tys.iter().map(|ty| Value::new(ty.clone())).collect::<Vec<Value>>());
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid tuple value: {}", val_str);
                    Value::new(ty.clone())
                }
            },
            TypesKind::Func(tys, rty) => {
                if Value::is_ptr(val_str) {
                    // Deal with value string: `&1243`
                    Value::fill_string(val_str, Some(Types::ptr(Types::func(tys.clone(), rty.clone()))))
                } else if Value::is_ident(val_str) {
                    let mut val = Value::new(Types::ptr(Types::func(tys.clone(), rty.clone())));
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid func name: {}", val_str);
                    Value::new(ty.clone())
                }
            },
            TypesKind::Struct(tys) => {
                if Value::is_ptr(val_str) {
                    Value::fill_string(val_str, Some(Types::ptr(Types::stc(tys.clone()))))
                } else if Value::is_ident(val_str) {
                    let mut val = Value::new(Types::ptr(Types::stc(tys.clone())));
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid struct name: {}", val_str);
                    Value::new(ty.clone())
                }
            },
            TypesKind::Ptr(ty) => {
                if Value::is_ptr(val_str) {
                    Value::ptr(Some(Value::fill_string(val_str, None)), Some(Value::new(ty.clone())))
                } else if Value::is_ptr_ref(val_str) {
                    // del `&`
                    let val_str = val_str[1..val_str.len()].trim();
                    Value::ptr(None, Some(Value::fill_string(val_str, Some(ty.clone()))))
                } else if Value::is_ident(val_str) {
                    let mut val = Value::new(Types::ptr(ty.clone()));
                    val.set_name(val_str.to_string());
                    val
                } else {
                    log_error!("Invalid ptr value: {}", val_str);
                    Value::new(ty.clone())
                }
            }
            _ => {
                log_error!("Invalid value: {}", val_str);
                Value::new(ty)
            }
        }
    }

    /// change new Value to Self
    pub fn change(&mut self, value: Value) {
        *self = value;
    }

    /// get ptr ref val
    pub fn get_ptr_ref(&self) -> Value {
        if !self.is_type_ptr() || self.ref_val.is_none() {
            log_warning!("Value has no ref");
            Value::new(Types::none())
        } else {
            self.ref_val.as_ref().unwrap().borrow().clone()
        }
    }


    /// Append Value at tail Align by u8 version: you will reserve scale info
    pub fn append(&mut self, value: Value) -> &mut Value {
        // 1. Change the Kind to tuple: (self, value)
        self.set_kind(TypesKind::Tuple(vec![self.ty.clone(), value.ty.clone()]));
        // 2. Extend the val vec
        self.val.borrow_mut().extend_from_slice(&value.val.borrow());
        // 3. Check the size
        assert_eq!(self.size(), self.val.borrow().len());
        // 4. set align mode
        self.set_align(true);
        self
    }

    /// Insert Value at head Align by u8 version
    pub fn insert(&mut self, value: Value) -> &mut Value{
        let mut new_val = value;
        new_val.append(self.clone());
        *self = new_val;
        self
    }

    /// Concat Value No Align: you will lost scale info
    pub fn concat(&mut self, value: Value) -> &mut Value {
        // 1. Change the Kind to array u8
        let l_scale = self.scale_sum();
        let r_scale = value.scale_sum();
        let arr_len = (l_scale + r_scale) / 8 + ((l_scale + r_scale) % 8 > 0) as usize;
        let sum_scale = l_scale + r_scale;
        let is_bits = self.ty.kind().is_bit() || value.ty.kind().is_bit();
        let l_extra_bits = l_scale % 8;
        let l_arr_len = l_scale / 8 + (l_scale % 8 > 0) as usize;
        let mask = (1 << l_extra_bits) - 1;
        // 2. Pop self val vec util len == l_arr_len
        while self.val.borrow().len() > l_arr_len {
            self.val.borrow_mut().pop();
        }
        // 3. Concat the val vec
        if l_extra_bits > 0 {
            let mut l_byte : u8;
            let mut r_byte : u8;
            for i in 0..value.val.borrow().len() {
                if i == 0 {
                    // first time read must: l[3:1] + r[5:1] -> l[8:4]
                    l_byte = self.val.borrow_mut().pop().unwrap();
                    r_byte = value.val.borrow()[i].clone();
                } else {
                    // next time read must: l[8:6] -> l[3:1] + r[5:1] -> l[8:4]
                    l_byte = value.val.borrow()[i - 1].clone() >> (8 - l_extra_bits);
                    r_byte = value.val.borrow()[i].clone();
                }
                l_byte = l_byte & mask;
                r_byte = r_byte << l_extra_bits;
                self.val.borrow_mut().push(l_byte | r_byte);
            }
        } else {
            self.val.borrow_mut().extend_from_slice(&value.val.borrow());
        }
        // Change type
        if is_bits {
            self.set_kind(TypesKind::Bit(sum_scale));
        } else {
            self.set_kind(TypesKind::Array(Types::u8(), arr_len));
        }
        // 4. Check the size
        assert_eq!(self.size(), self.val.borrow().len());
        // 5. set align mode
        self.set_align(false);
        self
    }

    /// Resize Value to scale (if scale > self.scale_sum: extend with 0)
    pub fn resize(&mut self, scale: usize) -> &mut Value {
        self.val.borrow_mut().resize(scale / 8 + (scale % 8 > 0) as usize, 0);
        // set kind
        self.set_kind(TypesKind::Array(Types::u8(), scale / 8 + (scale % 8 > 0) as usize));
        assert_eq!(self.scale_sum(), scale);
        self
    }

    /// Divide Value -> Vec<Value> by scale
    pub fn divide(&self) -> Vec<Value> {
        let mut res = Vec::new();
        let scale_vec = self.scale();
        let mut scale_sum = 0;
        let mut idx = 0;
        for i in 0..scale_vec.len() {
            res.push(self.get(idx, scale_vec[i]));
            scale_sum += scale_vec[i];
            idx += scale_sum / 8 + if scale_sum % 8 > 0 { 1 } else { 0 };
        }
        res
    }

    /// Set Value use vec<8> by index
    pub fn set(&mut self, index: usize, value: Value) {
        self.bound(index, value.scale_sum());
        self.val.borrow_mut()[index..index + value.size()].copy_from_slice(&value.val.borrow());
    }
    
    /// Get IRValue from vec<u8> by index and scale
    pub fn get(&self, index: usize, scale: usize) -> Value {
        self.bound(index, scale);
        let offset = scale / 8;
        let addition = scale % 8;
        let is_full = addition > 0;
        let cap = offset + if is_full { 1 } else { 0 };
        let res = Value::new(Types::array(Types::u8(), cap));
        for i in 0..offset {
            res.val.borrow_mut()[i] = self.val.borrow()[index + i];
        }
        if is_full {
            res.val.borrow_mut()[offset] = self.val.borrow()[index + offset] >> (8 - addition);
        }
        res
    }

    /// check name
    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    /// set name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// get name
    pub fn name(&self) -> String {
        if self.name.is_some() {
            self.name.as_ref().unwrap().clone()
        } else {
            log_warning!("Value has no name");
            "(none)".to_string()
        }
    }

    /// just change value vec, must size equal
    pub fn set_val(&mut self, value: Value) {
        assert_eq!(value.size(), self.size());
        self.val = value.val;
    }

    /// just change value vec and mutable
    pub fn set_val_mut(&mut self, value: Value) {
        self.val = value.val;
    }

    /// set scale
    pub fn set_scale(&mut self, scale: usize) {
        // copy old val: extend or cut
        if scale > self.val.borrow().len() * 8 {
            self.val.borrow_mut().extend_from_slice(&[0; 8]);
        } else {
            self.val.borrow_mut().truncate(scale / 8 + if scale % 8 > 0 { 1 } else { 0 });
        }
        // set new scale
        self.set_kind(TypesKind::Array(Types::u8(), scale / 8 + if scale % 8 > 0 { 1 } else { 0 }));
        assert!(self.scale_sum() == scale);
    }

    /// Set align mode: if set this use scale to search
    pub fn set_align(&mut self, is_strict_align: bool) {
        if is_strict_align {
            self.flag |= 0x0010;
        } else {
            self.flag &= 0xffef;
        }
    }

    /// is strict align
    pub fn is_strict_align(&self) -> bool {
        (self.flag & 0x0010) != 0
    }

    pub fn is_type_ptr(&self) -> bool {
        if let TypesKind::Ptr(_) = self.kind() {
            true
        } else {
            false
        }
    }

    // ==================== Value.ctl ==================== //

    /// Get the kind of the Value
    pub fn kind(&self) -> &TypesKind {
        self.ty.kind()
    }

    /// Get the size of the Value
    pub fn size(&self) -> usize {
        self.ty.size()
    }

    /// Get the scale of the Value
    pub fn scale(&self) -> Vec<usize> {
        self.ty.scale()
    }

    /// Get the sum of the scale of the Value
    pub fn scale_sum(&self) -> usize {
        self.ty.scale().iter().sum()
    }

    /// Get the bits hex string of the Value: Default little-endian
    /// - `index`: start index
    /// - `byte_num`: number of bytes you want, -1 means all
    pub fn hex(&self, index: usize, byte_num: i32, big_endian: bool) -> String {
        let mut num = byte_num as usize;
        // Check index and byte_num
        if byte_num < 0 {
            num = self.size() - index;
        }
        // Get index byte hexs
        let mut hexs = String::new();
        if big_endian {
            hexs.push_str("0X");
            for i in 0..num {
                hexs.push_str(&format!("{:02X}", self.get_byte(index + num - 1 - i as usize)));
                if i < num - 1 {
                    hexs.push(' ');
                }
            }
        } else {
            hexs.push_str("0x");
            for i in 0..num {
                hexs.push_str(&format!("{:02x}", self.get_byte(index + i as usize)));
                if i < num - 1 {
                    hexs.push(' ');
                }
            }
        }
        hexs
    }

    /// Get the bits binary string of the Value: Default little-endian
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
        if big_endian {
            bin.push_str("0B");
            for i in 0..num {
                bin.push_str(&format!("{:08b}", self.get_byte(index + num - 1 - i as usize)));
                if i < num - 1 {
                    bin.push(' ');
                }
            }
        } else {
            bin.push_str("0b");
            for i in 0..num {
                bin.push_str(&format!("{:08b}", self.get_byte(index + i as usize)));
                if i < num - 1 {
                    bin.push(' ');
                }
            }
        }
        bin
    }

    /// Scale version bin
    pub fn bin_scale(&self, index: usize, byte_num: i32, big_endian: bool) -> String {
        let mut num = byte_num as usize;
        // Check index and byte_num
        if byte_num < 0 {
            num = self.size() - index;
        }
        // Get index byte bin
        let mut bin = String::new();
        if big_endian {
            bin.push_str("0B");
            for i in 0..num {
                if (i+1) * 8 > self.scale_sum() {
                    let addition = self.scale_sum() % 8;
                    bin.insert_str(2, &format!("{:08b}", 
                        self.get_byte(index + i as usize))
                        .chars().rev().take(addition).collect::<String>().chars().rev().collect::<String>()
                    );
                    break;
                }
                bin.insert_str(2, &format!("{:08b}", self.get_byte(index + i as usize)));
                if i < num - 1 {
                    bin.insert(2, ' ');
                }
            }
        } else {
            bin.push_str("0b");
            for i in 0..num {
                if (i+1) * 8 > self.scale_sum() {
                    let addition = self.scale_sum() % 8;
                    bin.push_str(&format!("{:08b}", 
                        self.get_byte(index + i as usize))
                        .chars().rev().take(addition).collect::<String>().chars().rev().collect::<String>()
                    );
                    break;
                }
                bin.push_str(&format!("{:08b}", self.get_byte(index + i as usize)));
                if i < num - 1 {
                    bin.push(' ');
                }
            }
        }
        bin
    }

    /// Check Value size bound
    pub fn bound(&self, index: usize, scale : usize) {
        // index + size of T should <= self.size()
        let left = index * 8 + scale;
        let mut right = self.size() * 8;
        let scale = self.scale_sum();
        let mut use_scale = false;
        if scale < right {
            right = scale;
            use_scale = true; 
        }
        let is_valid = left <= right;
        let sym;
        if use_scale == true {
            sym = "scale".to_string();
        } else {
            sym = "wsize".to_string();
        }
        if !is_valid {
            log_fatal!("Index out of scale bounds: idx+val: {} > {}: {}", left, sym, right);
        }
    }

    // ==================== Value.get ==================== //

    /// Get value by byte: no bound fill 0
    pub fn get_byte(&self, index: usize) -> u8 {
        let buffer = self.val.borrow();
        if buffer.len() < index + 1 {
            // log_warning!("Byte index out of bounds: {}", index);
            // get val clone , resize and fill 0
            let mut val = self.val.borrow().clone();
            val.resize(index + 1, 0);
            return val[index];
        }
        buffer[index]
    }

    /// Get value by half: no bound
    pub fn get_half(&self, index: usize) -> u16 {
        let buffer = self.val.borrow();
        if buffer.len() < index + 2 {
            // log_warning!("Half index out of bounds: {}", index);
            // get val clone , resize and fill 0
            let mut val = self.val.borrow().clone();
            val.resize(index + 2, 0);
            return u16::from_le_bytes([val[index], val[index + 1]]);
        }
        u16::from_le_bytes([buffer[index], buffer[index + 1]])
    }

    /// Get value by word: no bound
    pub fn get_word(&self, index: usize) -> u32 {
        let buffer = self.val.borrow();
        if buffer.len() < index + 4 {
            // log_warning!("Word index out of bounds: {}", index);
            // get val clone , resize and fill 0
            let mut val = self.val.borrow().clone();
            val.resize(index + 4, 0);
            return u32::from_le_bytes([val[index], val[index + 1], val[index + 2], val[index + 3]]);
        }
        u32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by dword: no bound
    pub fn get_dword(&self, index: usize) -> u64 {
        let buffer = self.val.borrow();
        if buffer.len() < index + 8 {
            // log_warning!("Dword index out of bounds: {}", index);
            // get val clone , resize and fill 0
            let mut val = self.val.borrow().clone();
            val.resize(index + 8, 0);
            return u64::from_le_bytes([val[index], val[index + 1], val[index + 2], val[index + 3],
                val[index + 4], val[index + 5], val[index + 6], val[index + 7]]);
        }
        u64::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3],
            buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7]])
    }

    /// Get value by ubyte
    pub fn get_ubyte(&self, index: usize, offset: usize, scale: usize) -> u8 {
        let offset_bytes = offset / 8;
        let offset_bits = offset % 8;
        let mut scale = scale;
        // scale must <= 8
        if scale > 8 {
            scale = 8;
            log_error!("Scale must <= 8");
        }
        self.bound(index + offset_bytes, scale);
        let buffer = self.val.borrow();
        let res: u8;
        let flag = (offset_bits + scale) as i32 - 8;     // if > 0, have overflow
        // Offset bits and load value to res
        if offset_bits == 0 && scale == 8 {
            // nomal value `10001010`, offset `0`, scale `8` -> `10001010`
            res = buffer[index + offset_bytes] as u8;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 1 {
            // 0. get u16
            let buf_val : u16 = u16::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1]]);
            // 1. get mask
            let mask: u16 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u8;
        } else {
            // 0. get u8
            let buf_val : u8 = buffer[index + offset_bytes];
            // 1. get mask
            let mask: u8 = (1 << scale) - 1;
            // 2. get value
            res = (buf_val >> offset_bits) & mask;
        }
        res as u8
    }

    /// Get value by uhalf (u16 version)
    pub fn get_uhalf(&self, index: usize, offset: usize, scale: usize) -> u16 {
        let offset_bytes = offset / 8;
        let offset_bits = offset % 8;
        let mut scale = scale;
        // scale must <= 16
        if scale > 16 {
            scale = 16;
            log_error!("Scale must <= 16");
        }
        self.bound(index + offset_bytes, scale);
        if scale <= 8 {
            return self.get_ubyte(index, offset, scale) as u16
        }
        let buffer = self.val.borrow();
        let res: u16;
        let flag = (offset_bits + scale) as i32 - 8;     // if > 0, have overflow
        // Offset bits and load value to res
        if offset_bits == 0 && scale == 16 {
            // nomal value `10001010`, offset `0`, scale `16` -> `10001010`
            res = u16::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1]]);
        } else if flag > 0 && buffer.len() > index + offset_bytes + 3 {
            // 0. get u32
            let buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3]]);
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u16;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 2 {
            // 0. get u32
            let buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], 0]);
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u16;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 1 {
            // 0. get u16
            let buf_val : u16 = u16::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1]]);
            // 1. get mask
            let mask: u16 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u16;
        } else {
            // 0. get u8
            let buf_val : u8 = buffer[index + offset_bytes];
            // 1. get mask
            let mask: u8 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u16;
        }
        res
    }

    /// Get value by uword (u32 version)
    pub fn get_uword(&self, index: usize, offset: usize, scale: usize) -> u32 {
        let offset_bytes = offset / 8;
        let offset_bits = offset % 8;
        let mut scale = scale;
        // scale must <= 32
        if scale > 32 {
            scale = 32;
            log_error!("Scale must <= 32");
        }
        self.bound(index + offset_bytes, scale);
        if scale <= 16 {
            return self.get_uhalf(index, offset, scale) as u32
        }
        let buffer = self.val.borrow();
        let res: u32;
        let flag = (offset_bits + scale) as i32 - 8;     // if > 0, have overflow
        // Offset bits and load value to res
        if offset_bits == 0 && scale == 32 {
            // nomal value `10001010`, offset `0`, scale `32` -> `10001010`
            res = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3]]);
        } else if flag > 0 && buffer.len() > index + offset_bytes + 7 {
            // 0. get u64
            let buf_val : u64 = u64::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3],
                                                    buffer[index + offset_bytes + 4], buffer[index + offset_bytes + 5], buffer[index + offset_bytes + 6], buffer[index + offset_bytes + 7]]);
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 6 {
            // 0. get u64
            let buf_val : u64 = u64::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3],
                                                    buffer[index + offset_bytes + 4], buffer[index + offset_bytes + 5], buffer[index + offset_bytes + 6], 0]);
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 5 {
            // 0. get u64
            let buf_val : u64 = u64::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3],
                                                    buffer[index + offset_bytes + 4], buffer[index + offset_bytes + 5], 0, 0]);
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 4 {
            // 0. get u64
            let buf_val : u64 = u64::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3],
                                                    buffer[index + offset_bytes + 4], 0, 0, 0]);
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 3 {
            // 0. get u32
            let buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3]]);
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 2 {
            // 0. get u32
            let buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], 0]);
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else if flag > 0 && buffer.len() > index + offset_bytes + 1 {
            // 0. get u16
            let buf_val : u16 = u16::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1]]);
            // 1. get mask
            let mask: u16 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        } else{
            // 0. get u8
            let buf_val : u8 = u8::from_le_bytes([buffer[index + offset_bytes]]);
            // 1. get mask
            let mask: u8 = (1 << scale) - 1;
            // 2. get value
            res = ((buf_val >> offset_bits) & mask) as u32;
        }
        res
    }

    /// Get value by 8-bit
    pub fn get_u8(&self, index: usize) -> u8 {
        self.bound(index, 8);
        let buffer = self.val.borrow();
        u8::from_le_bytes([buffer[index]])
    }

    /// Get value by 16-bit
    pub fn get_u16(&self, index: usize) -> u16 {
        self.bound(index, 16);
        let buffer = self.val.borrow();
        u16::from_le_bytes([buffer[index], buffer[index + 1]])
    }

    /// Get value by 32-bit
    pub fn get_u32(&self, index: usize) -> u32 {
        self.bound(index, 32);
        let buffer = self.val.borrow();
        u32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by 64-bit
    pub fn get_u64(&self, index: usize) -> u64 {
        self.bound(index, 64);
        let buffer = self.val.borrow();
        u64::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7]])
    }

    /// Get value by 128-bit
    pub fn get_u128(&self, index: usize) -> u128 {
        self.bound(index, 128);
        let buffer = self.val.borrow();
        u128::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7],
            buffer[index + 8], buffer[index + 9], buffer[index + 10], buffer[index + 11], buffer[index + 12], buffer[index + 13], buffer[index + 14], buffer[index + 15]])
    }

    /// Get value by signed 8-bit
    pub fn get_i8(&self, index: usize) -> i8 {
        self.bound(index, 8);
        let buffer = self.val.borrow();
        i8::from_le_bytes([buffer[index]])
    }

    /// Get value by signed 16-bit
    pub fn get_i16(&self, index: usize) -> i16 {
        self.bound(index, 16);
        let buffer = self.val.borrow();
        i16::from_le_bytes([buffer[index], buffer[index + 1]])
    }

    /// Get value by signed 32-bit
    pub fn get_i32(&self, index: usize) -> i32 {
        self.bound(index, 32);
        let buffer = self.val.borrow();
        i32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by signed 64-bit
    pub fn get_i64(&self, index: usize) -> i64 {
        self.bound(index, 64);
        let buffer = self.val.borrow();
        i64::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7]])
    }

    /// Get value by signed 128-bit
    pub fn get_i128(&self, index: usize) -> i128 {
        self.bound(index, 128);
        let buffer = self.val.borrow();
        i128::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3], buffer[index + 4], buffer[index + 5], buffer[index + 6], buffer[index + 7],
            buffer[index + 8], buffer[index + 9], buffer[index + 10], buffer[index + 11], buffer[index + 12], buffer[index + 13], buffer[index + 14], buffer[index + 15]])
    }

    /// Get value by float
    pub fn get_f32(&self, index: usize) -> f32 {
        self.bound(index, 32);
        let buffer = self.val.borrow();
        f32::from_le_bytes([buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]])
    }

    /// Get value by double
    pub fn get_f64(&self, index: usize) -> f64 {
        self.bound(index, 64);
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


    // ==================== Value.set ==================== //

    /// Set type of the Value by `TypesKind`
    pub fn set_kind(&mut self, kind : TypesKind) {
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

    /// Set type of the Value by `Types`
    pub fn set_type(&mut self, ty : Types) {
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
        self.bound(index, 8);
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
            self.set_kind(TypesKind::Array(Types::u8(), val_size));
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

    /// Set hexs from string
    pub fn set_hexs(&mut self, index: usize, value: &str, change_length: bool, big_endian: bool) {
        let mut value = value.to_string();
        if value.len() % 2 != 0 {
            if big_endian {
                value.insert(0, '0');
            } else {
                value.push_str("0");
            }
        }
        // Set length
        if change_length {
            self.set_kind(TypesKind::Array(Types::u8(), (value.len() as f64 / 2.0).ceil() as usize));
        }
        // parser hexs from string like "eaff0000": evary time deal with 2 words
        // every two words be a u8 byte
        for i in 0..value.len() / 2 {
            // get two words
            let hexs :&str;
            if big_endian {
                hexs = &value[(value.len() - 2 - i * 2)..(value.len() - i * 2)];
            } else {
                hexs = &value[i * 2..i * 2 + 2];
            }
            // deal with two words
            self.set_8bit(index + i, u8::from_str_radix(hexs, 16).unwrap());
        }
    }

    /// Set value by unsigned byte.
    /// 
    /// `offset`: bit offset, from bytes' right side. 
    /// 
    /// `scale`(0~8): display width, from bytes' right side.
    /// 
    /// `value`: u8 value to set
    /// 
    /// Such as origin value `11111111 11111111`:
    /// 1. value `........ 10001010`, offset bits `2`, scale `3` -> `........ ...010..` -> `11111111 11101011`
    /// 2. value `........ 10001010`, offset bits `5`, scale `7` -> `...10001 010.....` -> `11110001 01011111`
    pub fn set_ubyte(&mut self, index: usize, offset: usize, scale: usize, value: u8) {
        let offset_bytes = offset / 8;
        let offset_bits = offset % 8;
        let mut scale = scale;
        // scale must <= 8
        if scale > 8 {
            scale = 8;
            log_error!("Scale must <= 8");
        }
        self.bound(index + offset_bytes, scale + offset_bits);
        // if log(value + 1) > scale, warning
        if value as u16 > ((1 as u16) << scale) - 1 {
            log_warning!("Value(u8) {} is too large for 2^{} - 1", value, scale);
        }
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        let flag = (offset_bits + scale) as i32 - 8;     // if > 0, have overflow
        // Offset bits and load value to buffer
        if offset_bits == 0 && scale == 8 {
            // nomal value `10001010`, offset `0`, scale `8` -> `10001010`
            buffer[index + offset_bytes] = bytes[0];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 1 {
            // 0. get u16
            let mut buf_val : u16 = u16::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1]]);
            let bytes_val : u16 = bytes[0] as u16;
            // 1. get mask
            let mask: u16 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
        } else {
            // 1. get mask
            let mask: u8 = (1 << scale) - 1;
            // 2. clear buffer[index + offset_bytes]
            buffer[index + offset_bytes] &= !(mask << offset_bits);
            // 3. set value
            buffer[index + offset_bytes] |= (bytes[0] & mask) << offset_bits;
        }
    }

    /// u16 version of set_ubyte
    pub fn set_uhalf(&mut self, index: usize, offset: usize, scale: usize, value: u16) {
        let offset_bytes = offset / 8;
        let offset_bits = offset % 8;
        let mut scale = scale;
        // scale must <= 16
        if scale > 16 {
            scale = 16;
            log_error!("Scale must <= 16");
        }
        self.bound(index + offset_bytes, scale + offset_bits);
        if scale <= 8 {
            self.set_ubyte(index, offset, scale, (value & 0xFF) as u8);
            return
        }
        // If log(value + 1) > scale, warning
        if value as u32 > ((1 as u32) << scale) - 1 {
            log_warning!("Value(u16) {} is too large for 2^{} - 1", value, scale);
        }
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        let flag = (offset_bits + scale) as i32 - 16;     // if > 0, have overflow
        // Offset bits and load value to buffer
        if offset_bits == 0 && scale == 16 {
            // nomal value `10001010`, offset `0`, scale `16` -> `10001010`
            buffer[index + offset_bytes] = bytes[0];
            buffer[index + offset_bytes + 1] = bytes[1];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 3 {
            // 0. get u32
            let mut buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], buffer[index + offset_bytes + 3]]);
            let bytes_val : u32 = value as u32;
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 2 {
            // 0. get u32
            let mut buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], buffer[index + offset_bytes + 2], 0]);
            let bytes_val : u32 = value as u32;
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 1 {
            // 0. get u24
            let mut buf_val : u32 = u32::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1], 0, 0]);
            let bytes_val : u32 = value as u32;
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
        } else {
            // 0. get u16
            let mut buf_val : u16 = u16::from_le_bytes([buffer[index + offset_bytes], buffer[index + offset_bytes + 1]]);
            let bytes_val : u16 = value as u16;
            // 1. get mask
            let mask: u16 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
        }

    }

    /// `set_uword`: u32 version of set_ubyte
    pub fn set_uword(&mut self, index: usize, offset: usize, scale: usize, value: u32) {
        let offset_bytes = offset / 8;
        let offset_bits = offset % 8;
        let mut scale = scale;
        // scale must <= 32
        if scale > 32 {
            scale = 32;
            log_error!("Scale must <= 32");
        }
        self.bound(index + offset_bytes, scale + offset_bits);
        if scale <= 16 {
            self.set_uhalf(index, offset, scale, (value & 0xFFFF) as u16);
            return
        }
        // if log(value + 1) > scale, warning
        if value as u64 > ((1 as u64) << scale) - 1 {
            log_warning!("Value(u32) {} is too large for 2^{} - 1", value, scale);
        }
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        let flag = (offset_bits + scale) as i32 - 32;     // if > 0, have overflow
        // Offset bits and load value to buffer
        if offset_bits == 0 && scale == 32 {
            // nomal value `10001010`, offset `0`, scale `32` -> `10001010`
            buffer[index + offset_bytes] = bytes[0];
            buffer[index + offset_bytes + 1] = bytes[1];
            buffer[index + offset_bytes + 2] = bytes[2];
            buffer[index + offset_bytes + 3] = bytes[3];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 7 {
            // 0. get u64
            let mut buf_val : u64 = u64::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2], 
                buffer[index + offset_bytes + 3], 
                buffer[index + offset_bytes + 4], 
                buffer[index + offset_bytes + 5], 
                buffer[index + offset_bytes + 6], 
                buffer[index + offset_bytes + 7]
            ]);
            let bytes_val : u64 = value as u64;
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
            buffer[index + offset_bytes + 4] = buf_bytes[4];
            buffer[index + offset_bytes + 5] = buf_bytes[5];
            buffer[index + offset_bytes + 6] = buf_bytes[6];
            buffer[index + offset_bytes + 7] = buf_bytes[7];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 6 {
            // 0. get u64
            let mut buf_val : u64 = u64::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2], 
                buffer[index + offset_bytes + 3], 
                buffer[index + offset_bytes + 4], 
                buffer[index + offset_bytes + 5], 
                buffer[index + offset_bytes + 6], 
                0
            ]);
            let bytes_val : u64 = value as u64;
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
            buffer[index + offset_bytes + 4] = buf_bytes[4];
            buffer[index + offset_bytes + 5] = buf_bytes[5];
            buffer[index + offset_bytes + 6] = buf_bytes[6];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 5 {
            // 0. get u64
            let mut buf_val : u64 = u64::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2], 
                buffer[index + offset_bytes + 3], 
                buffer[index + offset_bytes + 4], 
                buffer[index + offset_bytes + 5], 
                0, 
                0
            ]);
            let bytes_val : u64 = value as u64;
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
            buffer[index + offset_bytes + 4] = buf_bytes[4];
            buffer[index + offset_bytes + 5] = buf_bytes[5];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 4 {
            // 0. get u64
            let mut buf_val : u64 = u64::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2], 
                buffer[index + offset_bytes + 3], 
                buffer[index + offset_bytes + 4], 
                0, 
                0, 
                0
            ]);
            let bytes_val : u64 = value as u64;
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
            buffer[index + offset_bytes + 4] = buf_bytes[4];
        } else if flag > 0 && buffer.len() > index + offset_bytes + 3 {
            // 0. get u64
            let mut buf_val : u64 = u64::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2], 
                buffer[index + offset_bytes + 3], 
                0, 
                0, 
                0, 
                0
            ]);
            let bytes_val : u64 = value as u64;
            // 1. get mask
            let mask: u64 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
        } else if buffer.len() > index + offset_bytes + 2 {
            // 0. get u32
            let mut buf_val : u32 = u32::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2],
                0
            ]);
            let bytes_val : u32 = value as u32;
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
        } else if buffer.len() > index + offset_bytes + 1 {
            // 0. get u32
            let mut buf_val : u32 = u32::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                0,
                0
            ]);
            let bytes_val : u32 = value as u32;
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
        } else {
            // 0. get u32
            let mut buf_val : u32 = u32::from_le_bytes([
                buffer[index + offset_bytes], 
                buffer[index + offset_bytes + 1], 
                buffer[index + offset_bytes + 2],
                buffer[index + offset_bytes + 3]
            ]);
            let bytes_val : u32 = value as u32;
            // 1. get mask
            let mask: u32 = (1 << scale) - 1;
            // 2. clear buf_val
            buf_val &= !(mask << offset_bits);
            // 3. set value
            buf_val |= (bytes_val & mask) << offset_bits;
            // 4. set buf_val
            let buf_bytes = buf_val.to_le_bytes();
            buffer[index + offset_bytes] = buf_bytes[0];
            buffer[index + offset_bytes + 1] = buf_bytes[1];
            buffer[index + offset_bytes + 2] = buf_bytes[2];
            buffer[index + offset_bytes + 3] = buf_bytes[3];
        }
    }

    /// Set value by 8-bit: index by bytes and offset by bits
    pub fn set_8bit(&mut self, index: usize, value: u8) {
        self.bound(index, 8);
        let mut buffer = self.val.borrow_mut();
        buffer[index] = value;
    }

    /// Set value by 16-bit
    pub fn set_16bit(&mut self, index: usize, value: u16) {
        self.bound(index, 16);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
        buffer[index + 1] = bytes[1];
    }

    /// Set value by 24-bit
    pub fn set_24bit(&mut self, index: usize, value: u32) {
        self.bound(index, 24);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
        buffer[index + 1] = bytes[1];
        buffer[index + 2] = bytes[2];
    }

    /// Set value by 32-bit
    pub fn set_32bit(&mut self, index: usize, value: u32) {
        self.bound(index, 32);
        let mut buffer = self.val.borrow_mut();
        let bytes = value.to_le_bytes();
        buffer[index] = bytes[0];
        buffer[index + 1] = bytes[1];
        buffer[index + 2] = bytes[2];
        buffer[index + 3] = bytes[3];
    }

    /// Set value by 64-bit
    pub fn set_64bit(&mut self, index: usize, value: u64) {
        self.bound(index, 64);
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

    /// Set value by 128-bit
    pub fn set_128bit(&mut self, index: usize, value: u128) {
        self.bound(index, 128);
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
        buffer[index + 8] = bytes[8];
        buffer[index + 9] = bytes[9];
        buffer[index + 10] = bytes[10];
        buffer[index + 11] = bytes[11];
        buffer[index + 12] = bytes[12];
        buffer[index + 13] = bytes[13];
        buffer[index + 14] = bytes[14];
        buffer[index + 15] = bytes[15];
    }

    /// Set value by unsigned 8-bit
    pub fn set_u8(&mut self, index: usize, value: u8) {
        self.set_kind(TypesKind::U8);
        self.set_8bit(index, value);
    }

    /// Set value by unsigned 16-bit
    pub fn set_u16(&mut self, index: usize, value: u16) {
        self.set_kind(TypesKind::U16);
        self.set_16bit(index, value);
    }

    /// set value by unsigned 24-bit
    pub fn set_u24(&mut self, index: usize, value: u32) {
        self.set_kind(TypesKind::U24);
        self.set_24bit(index, value);
    }

    /// Set value by unsigned 32-bit
    pub fn set_u32(&mut self, index: usize, value: u32) {
        self.set_kind(TypesKind::U32);
        self.set_32bit(index, value);
    }

    /// Set value by unsigned 64-bit
    pub fn set_u64(&mut self, index: usize, value: u64) {
        self.set_kind(TypesKind::U64);
        self.set_64bit(index, value);
    }

    /// Set value by unsigned 128-bit
    pub fn set_u128(&mut self, index: usize, value: u128) {
        self.set_kind(TypesKind::U128);
        self.set_128bit(index, value);
    }

    /// Set value by signed 8-bit
    pub fn set_i8(&mut self, index: usize, value: i8) {
        self.set_kind(TypesKind::I8);
        self.set_8bit(index, value as u8);
    }

    /// Set value by signed 16-bit
    pub fn set_i16(&mut self, index: usize, value: i16) {
        self.set_kind(TypesKind::I16);
        self.set_16bit(index, value as u16);
    }

    /// Set value by signed 32-bit
    pub fn set_i32(&mut self, index: usize, value: i32) {
        self.set_kind(TypesKind::I32);
        self.set_32bit(index, value as u32);
    }

    /// Set value by signed 64-bit
    pub fn set_i64(&mut self, index: usize, value: i64) {
        self.set_kind(TypesKind::I64);
        self.set_64bit(index, value as u64);
    }

    /// Set value by signed 128-bit
    pub fn set_i128(&mut self, index: usize, value: i128) {
        self.set_kind(TypesKind::I128);
        self.set_128bit(index, value as u128);
    }

    /// Set value by float
    pub fn set_f32(&mut self, index: usize, value: f32) {
        self.set_kind(TypesKind::F32);
        let bytes = value.to_le_bytes();
        self.set_8bit(index, bytes[0]);
        self.set_8bit(index + 1, bytes[1]);
        self.set_8bit(index + 2, bytes[2]);
        self.set_8bit(index + 3, bytes[3]);
    }

    /// Set value by double
    pub fn set_f64(&mut self, index: usize, value: f64) {
        self.set_kind(TypesKind::F64);
        let bytes = value.to_le_bytes();
        self.set_8bit(index, bytes[0]);
        self.set_8bit(index + 1,bytes[1]);
        self.set_8bit(index + 2,bytes[2]);
        self.set_8bit(index + 3,bytes[3]);
        self.set_8bit(index + 4,bytes[4]);
        self.set_8bit(index + 5,bytes[5]);
        self.set_8bit(index + 6,bytes[6]);
        self.set_8bit(index + 7,bytes[7]);
    }

    /// Set value by array
    pub fn set_array(&mut self, value: Vec<Value>) {
        self.set_kind(TypesKind::Array(value[0].ty.clone(), value.len()));
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
    pub fn set_tuple(&mut self, value: Vec<Value>) {
        // Get Vec<Types> from Vec<Value>
        let mut types = Vec::new();
        let mut types_size = 0;
        for val in value.iter() {
            types.push(val.ty.clone());
            types_size += val.size();
        }
        self.set_kind(TypesKind::Tuple(types));
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

    /// Set pointer value
    pub fn set_ptr(&mut self, value: Value) {
        // cover val vec
        let mut value = value;
        let size = self.size();
        self.set(0, value.resize(size * 8).clone());
        assert_eq!(self.size(), size);
    }

    /// Set pointer reference value
    pub fn set_ptr_ref(&mut self, value: Value) {
        self.set_kind(TypesKind::Ptr(value.ty.clone()));
        self.ref_val = Some(Rc::new(RefCell::new(value)));
    }

    /// Set value by char(4-bits)
    pub fn set_char(&mut self, index: usize, value: char) {
        self.set_8bit(index, value as u8);
    }

    /// Set value by string
    pub fn set_str(&mut self, value: &str) {
        self.set_kind(TypesKind::Array(Types::u8(), value.len()));
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

    // ==================== Value.is ===================== //

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

    /// Check str if the value is u128 number
    pub fn is_u128(str : &str) -> bool {
        str.parse::<u128>().is_ok()
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

    /// Check str if the value is i128 number
    pub fn is_i128(str : &str) -> bool {
        str.parse::<i128>().is_ok()
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

    /// Check str if the value is ident
    pub fn is_ident(str : &str) -> bool {
        // digit or letter or `_`: can't start with digit
        str.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !str.starts_with(|c: char| c.is_ascii_digit())
    }

    /// Check str if the value is pointer: &basictype / struct / function
    pub fn is_ptr(str : &str) -> bool {
        Self::is_u8(str) || Self::is_u16(str) || Self::is_u32(str) || Self::is_u64(str) || Self::is_u128(str)
        || Self::is_i8(str) || Self::is_i16(str) || Self::is_i32(str) || Self::is_i64(str) || Self::is_i128(str)
        || Self::is_hex(str) || Self::is_bin(str) || Self::is_array(str) || Self::is_str(str)
    }

    pub fn is_ptr_ref(str : &str) -> bool {
        let mut str = str.trim();
        let mut is_ref = false;
        if str.starts_with("&") {
            str = str[1..].trim();
            is_ref = Self::is_u8(str) || Self::is_u16(str) || Self::is_u32(str) || Self::is_u64(str) || Self::is_u128(str)
            || Self::is_i8(str) || Self::is_i16(str) || Self::is_i32(str) || Self::is_i64(str) || Self::is_i128(str)
            || Self::is_f32(str) || Self::is_f64(str) || Self::is_hex(str) || Self::is_bin(str) || Self::is_array(str) 
            || Self::is_str(str) || Self::is_tuple(str) || Self::is_ptr_ref(str);
        }
        is_ref
    }

    /// Check str if the value is hex
    pub fn is_hex(str : &str) -> bool {
        // begin with `0x` or `0X`
        str.starts_with("0x") || str.starts_with("0X")
    }

    /// Check str if the value is bin
    pub fn is_bin(str : &str) -> bool {
        // begin with `0b` or `0B`
        str.starts_with("0b") || str.starts_with("0B")
    }
    

    // ==================== Value.from =================== //

    /// bits filter: `0b00001010 00001010` -> `0000101000001010` and `is_big_endian`
    pub fn bits_filter(value: &str) -> (String, bool) {
        let is_big_endian = value.starts_with("0B");
        let value = value[2..].to_string();
        let value = value.trim();
        let mut new_val = String::new();
        for i in 0..value.len() {
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
        (new_val, is_big_endian)
    }

    /// Get bits from String: `0b00001010 00001010`
    pub fn bits(value: &str) -> Value {
        if value.is_empty() {
            return Value::new(Types::bit(0));
        }
        let (new_val, is_big_endian) = Self::bits_filter(value);
        // Get val size: Upper bound
        let val_size = (new_val.len() as f64 / 8.0).ceil() as usize;
        let mut val = Value::new(Types::array(Types::u8(), val_size));

        // Set value
        val.set_bits(0, &new_val, true, is_big_endian);
        val
    }

    /// hex filter: `0xea ff 00 00` -> `eaff0000`
    pub fn hex_filter(value: &str) -> (String, bool) {
        let is_big_endian = value.starts_with("0X");
        let value = value[2..].to_string();
        let value = value.trim();
        let mut new_val = String::new();
        for i in 0..value.len() {
            // deal with hex value like `ea` `ff` `00` `00`
            // if is hex num
            if value.chars().nth(i).unwrap().is_ascii_hexdigit() {
                new_val += value.chars().nth(i).unwrap().to_string().as_str();
            } else {
                continue;
            }
        }
        (new_val, is_big_endian)
    }

    /// Get hex from String: `0xea ff 00 00`
    pub fn hexs(value: &str) -> Value {
        let (new_val, is_big_endian) = Self::hex_filter(value);
        let mut val = Value::new(Types::array(Types::u8(), new_val.len() / 2));
        val.set_hexs(0, &new_val, true, is_big_endian);
        val
    }


    /// Get value from Bit: scale & u128
    pub fn ubit(scale : usize, value: u128) -> Value {
        let mut val = Value::new(Types::bit(scale));
        // Get Vec<u8> from u128 and get first val.size item
        let mut bit_vec = u128::to_le_bytes(value).to_vec();
        bit_vec.resize(val.size(), 0);
        // Set Scale Addition, otherwise 0
        let offset_bytes = scale / 8;
        let offset_bits = scale % 8;
        if offset_bits != 0 {
            bit_vec[offset_bytes] &= (1 << offset_bits) - 1;
        }
        val.val = RefCell::new(bit_vec);
        val
    }

    /// Get Value from Bit: scale & i128
    pub fn ibit(scale : usize, value: i128) -> Value {
        let mut val = Value::new(Types::bit(scale));
        // Get Vec<u8> from i128 and get first val.size item
        let mut bit_vec = i128::to_le_bytes(value).to_vec();
        bit_vec.resize(val.size(), 0);
        // Set Scale Addition, otherwise 0
        let offset_bytes = scale / 8;
        let offset_bits = scale % 8;
        if offset_bits != 0 {
            bit_vec[offset_bytes] &= (1 << offset_bits) - 1;
        }
        val.val = RefCell::new(bit_vec);
        val
    }

    /// Get value from Bit: scale & i128
    pub fn bit(scale : usize, value: i128) -> Value {
        Value::ibit(scale, value)
    }

    /// Get value from u8 -> u1
    pub fn u1(value: u8) -> Value {
        let mut val = Value::new(Types::u1());
        val.set_ubyte(0,0, 1, value);
        val
    }

    /// Get value from u8 -> u2
    pub fn u2(value: u8) -> Value {
        let mut val = Value::new(Types::u2());
        val.set_ubyte(0,0, 2, value);
        val
    }

    /// Get value from u8 -> u3
    pub fn u3(value: u8) -> Value {
        let mut val = Value::new(Types::u3());
        val.set_ubyte(0,0, 3, value);
        val
    }

    /// Get value from u8 -> u4
    pub fn u4(value: u8) -> Value {
        let mut val = Value::new(Types::u4());
        val.set_ubyte(0,0, 4, value);
        val
    }

    /// Get value from u8 -> u5
    pub fn u5(value: u8) -> Value {
        let mut val = Value::new(Types::u5());
        val.set_ubyte(0,0, 5, value);
        val
    }

    /// Get value from u8 -> u6
    pub fn u6(value: u8) -> Value {
        let mut val = Value::new(Types::u6());
        val.set_ubyte(0,0, 6, value);
        val
    }

    /// Get value from u8 -> u7
    pub fn u7(value: u8) -> Value {
        let mut val = Value::new(Types::u7());
        val.set_ubyte(0,0, 7, value);
        val
    }

    /// Get value from u8
    pub fn u8(value: u8) -> Value {
        let mut val = Value::new(Types::u8());
        val.set_u8(0, value);
        val
    }

    /// Get value from u16 -> u9
    pub fn u9(value: u16) -> Value {
        let mut val = Value::new(Types::u9());
        val.set_uhalf(0, 0, 9, value);
        val
    }

    /// Get value from u16 -> u10
    pub fn u10(value: u16) -> Value {
        let mut val = Value::new(Types::u10());
        val.set_uhalf(0, 0, 10, value);
        val
    }

    /// Get value from u16 -> u11
    pub fn u11(value: u16) -> Value {
        let mut val = Value::new(Types::u11());
        val.set_uhalf(0, 0, 11, value);
        val
    }

    /// Get value from u16 -> u12
    pub fn u12(value: u16) -> Value {
        let mut val = Value::new(Types::u12());
        val.set_uhalf(0, 0, 12, value);
        val
    }

    /// Get value from u16 -> u13
    pub fn u13(value: u16) -> Value {
        let mut val = Value::new(Types::u13());
        val.set_uhalf(0, 0, 13, value);
        val
    }

    /// Get value from u16 -> u14
    pub fn u14(value: u16) -> Value {
        let mut val = Value::new(Types::u14());
        val.set_uhalf(0, 0, 14, value);
        val
    }

    /// Get value from u16 -> u15
    pub fn u15(value: u16) -> Value {
        let mut val = Value::new(Types::u15());
        val.set_uhalf(0, 0, 15, value);
        val
    }

    /// Get value from u16
    pub fn u16(value: u16) -> Value {
        let mut val = Value::new(Types::u16());
        val.set_u16(0, value);
        val
    }

    /// Get value from u17
    pub fn u17(value: u32) -> Value {
        let mut val = Value::new(Types::u17());
        val.set_uword(0, 0, 17, value);
        val
    }

    /// Get value from u18
    pub fn u18(value: u32) -> Value {
        let mut val = Value::new(Types::u18());
        val.set_uword(0, 0, 18, value);
        val
    }

    /// Get value from u19
    pub fn u19(value: u32) -> Value {
        let mut val = Value::new(Types::u19());
        val.set_uword(0, 0, 19, value);
        val
    }

    /// Get value from u20
    pub fn u20(value: u32) -> Value {
        let mut val = Value::new(Types::u20());
        val.set_uword(0, 0, 20, value);
        val
    }

    /// Get value from u21
    pub fn u21(value: u32) -> Value {
        let mut val = Value::new(Types::u21());
        val.set_uword(0, 0, 21, value);
        val
    }

    /// Get value from u22
    pub fn u22(value: u32) -> Value {
        let mut val = Value::new(Types::u22());
        val.set_uword(0, 0, 22, value);
        val
    }

    /// Get value from u23
    pub fn u23(value: u32) -> Value {
        let mut val = Value::new(Types::u23());
        val.set_uword(0, 0, 23, value);
        val
    }

    /// Get value from u24
    pub fn u24(value: u32) -> Value {
        let mut val = Value::new(Types::u24());
        val.set_uword(0, 0, 24, value);
        val
    }

    /// Get value from u25
    pub fn u25(value: u32) -> Value {
        let mut val = Value::new(Types::u25());
        val.set_uword(0, 0, 25, value);
        val
    }

    /// Get value from u26
    pub fn u26(value: u32) -> Value {
        let mut val = Value::new(Types::u26());
        val.set_uword(0, 0, 26, value);
        val
    }

    /// Get value from u27
    pub fn u27(value: u32) -> Value {
        let mut val = Value::new(Types::u27());
        val.set_uword(0, 0, 27, value);
        val
    }

    /// Get value from u28
    pub fn u28(value: u32) -> Value {
        let mut val = Value::new(Types::u28());
        val.set_uword(0, 0, 28, value);
        val
    }

    /// Get value from u29
    pub fn u29(value: u32) -> Value {
        let mut val = Value::new(Types::u29());
        val.set_uword(0, 0, 29, value);
        val
    }

    /// Get value from u30
    pub fn u30(value: u32) -> Value {
        let mut val = Value::new(Types::u30());
        val.set_uword(0, 0, 30, value);
        val
    }

    /// Get value from u31
    pub fn u31(value: u32) -> Value {
        let mut val = Value::new(Types::u31());
        val.set_uword(0, 0, 31, value);
        val
    }


    /// Get value from u32
    pub fn u32(value: u32) -> Value {
        let mut val = Value::new(Types::u32());
        val.set_u32(0, value);
        val
    }

    /// Get value from u64
    pub fn u64(value: u64) -> Value {
        let mut val = Value::new(Types::u64());
        val.set_u64(0, value);
        val
    }

    /// Get value from u128
    pub fn u128(value: u128) -> Value {
        let mut val = Value::new(Types::u128());
        val.set_u128(0, value);
        val
    }

    /// Get value from i8
    pub fn i8(value: i8) -> Value {
        let mut val = Value::new(Types::i8());
        val.set_i8(0, value);
        val
    }

    /// Get value from i16
    pub fn i16(value: i16) -> Value {
        let mut val = Value::new(Types::i16());
        val.set_i16(0, value);
        val
    }

    /// Get value from i32
    pub fn i32(value: i32) -> Value {
        let mut val = Value::new(Types::i32());
        val.set_i32(0, value);
        val
    }

    /// Get value from i64
    pub fn i64(value: i64) -> Value {
        let mut val = Value::new(Types::i64());
        val.set_i64(0, value);
        val
    }

    /// Get value from i128
    pub fn i128(value: i128) -> Value {
        let mut val = Value::new(Types::i128());
        val.set_i128(0, value);
        val
    }

    /// Get value from float
    pub fn f32(value: f32) -> Value {
        let mut val = Value::new(Types::f32());
        val.set_f32(0, value);
        val
    }

    /// Get value from double
    pub fn f64(value: f64) -> Value {
        let mut val = Value::new(Types::f64());
        val.set_f64(0, value);
        val
    }

    /// Get value from array u8
    pub fn array_u8(value: RefCell<Vec<u8>>) -> Value {
        // Set kind
        let mut val = Value::new(Types::array(Types::u8(), value.borrow().len()));
        // Set local Vec to new Vec
        val.val = value;
        val
    }

    /// Get value from array
    pub fn array(value: Vec<Value>) -> Value {
        // Set kind
        let mut val = Value::new(Types::array(value[0].ty.clone(), value.len()));
        val.set_array(value);
        val
    }

    /// Get value from tuple
    pub fn tuple(value: Vec<Value>) -> Value {
        // Get Vec<Types> from Vec<Value>
        let mut types = Vec::new();
        for val in value.iter() {
            types.push(val.ty.clone());
        }
        // Set kind
        let mut val = Value::new(Types::tuple(types));
        val.set_tuple(value);
        val
    }

    /// Get value from pointer
    pub fn ptr(ptr_val: Option<Value>, ref_val: Option<Value>) -> Value {
        let mut val= Value::new(Types::ptr(Types::void()));
        if ptr_val.is_some() {
            val.set_ptr(ptr_val.unwrap());
        }
        if ref_val.is_some() {
            val.set_ptr_ref(ref_val.unwrap());
        }
        val
    }

    /// Get value from text string -> array: `"hello"`
    pub fn str(value: &str) -> Value {
        let mut val = Value::new(Types::array(Types::u8(), value.len()));
        val.set_str(value);
        val
    }

    /// Get value from string
    pub fn from_string(val_str: &str) -> Value {
        let mut val_str = val_str.trim();
        // 1. Deal with `value: types`, check type behind `:`
        // if exist type, get kind of type
        let res: Value;
        let ty = if val_str.contains(":") {
            let parts = val_str.splitn(2, ":").collect::<Vec<&str>>();
            val_str = parts[0].trim();
            let ty_str = parts[1].trim();
            let ty = Types::from_string(ty_str);
            Some(ty)
        } else {
            None
        };

        res = Value::fill_string(val_str, ty);
        res
    }

}

impl Default for Value {
    /// Default value type is i32
    fn default() -> Self {
        Self {
            name: None,
            ty: Types::void(),
            val: RefCell::new(Vec::new()),
            ref_val: None,
            flag: 0
        }
    }
}

impl Display for Value {
    /// Display the Value by Types
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            TypesKind::U1 => write!(f, "{}", self.get_ubyte(0, 0, 1)),
            TypesKind::U2 => write!(f, "{}", self.get_ubyte(0, 0, 2)),
            TypesKind::U3 => write!(f, "{}", self.get_ubyte(0, 0, 3)),
            TypesKind::U4 => write!(f, "{}", self.get_ubyte(0, 0, 4)),
            TypesKind::U5 => write!(f, "{}", self.get_ubyte(0, 0, 5)),
            TypesKind::U6 => write!(f, "{}", self.get_ubyte(0, 0, 6)),
            TypesKind::U7 => write!(f, "{}", self.get_ubyte(0, 0, 7)),
            TypesKind::U8 => write!(f, "{}", self.get_u8(0)),
            TypesKind::U9 => write!(f, "{}", self.get_uhalf(0, 0, 9)),
            TypesKind::U10 => write!(f, "{}", self.get_uhalf(0, 0, 10)),
            TypesKind::U11 => write!(f, "{}", self.get_uhalf(0, 0, 11)),
            TypesKind::U12 => write!(f, "{}", self.get_uhalf(0, 0, 12)),
            TypesKind::U13 => write!(f, "{}", self.get_uhalf(0, 0, 13)),
            TypesKind::U14 => write!(f, "{}", self.get_uhalf(0, 0, 14)),
            TypesKind::U15 => write!(f, "{}", self.get_uhalf(0, 0, 15)),
            TypesKind::U16 => write!(f, "{}", self.get_u16(0)),
            TypesKind::U32 => write!(f, "{}", self.get_u32(0)),
            TypesKind::U64 => write!(f, "{}", self.get_u64(0)),
            TypesKind::I8 => write!(f, "{}", self.get_i8(0)),
            TypesKind::I16 => write!(f, "{}", self.get_i16(0)),
            TypesKind::I32 => write!(f, "{}", self.get_i32(0)),
            TypesKind::I64 => write!(f, "{}", self.get_i64(0)),
            TypesKind::F32 => write!(f, "{}", self.get_f32(0)),
            TypesKind::F64 => write!(f, "{}", self.get_f64(0)),
            _ => write!(f, "{}", self.hex(0, -1, false)),
        }
    }
}

impl Eq for Value {}



// ============================================================================== //
//                               Unit Tests
// ============================================================================== //


#[cfg(test)]
mod val_test {

    use super::*;

    #[test]
    fn val_get() {
        let val = Value::u8(12 as u8);
        assert_eq!(val.get_u8(0), 12 as u8);
        let val = Value::i8(-12 as i8);
        assert_eq!(val.get_i8(0), -12 as i8);
        let val = Value::u16(12 as u16);
        assert_eq!(val.get_u16(0), 12 as u16);
        let val = Value::i16(-12 as i16);
        assert_eq!(val.get_i16(0), -12 as i16);
        let val = Value::u32(12 as u32);
        assert_eq!(val.get_u32(0), 12 as u32);
        let val = Value::i32(-12 as i32);
        assert_eq!(val.get_i32(0), -12 as i32);
        let val = Value::u64(129 as u64);
        assert_eq!(val.get_u64(0), 129 as u64);
        let val = Value::i64(-12 as i64);
        assert_eq!(val.get_i64(0), -12 as i64);
        let val = Value::f32(12.33f32);
        assert_eq!(val.get_f32(0), 12.33f32);
        let val = Value::f64(12.34f64);
        assert_eq!(val.get_f64(0), 12.34f64);
    }

    #[test]
    fn val_print() {
        let mut val = Value::new(Types::u8());
        val.set_u8(0, 9 as u8);
        assert_eq!(val.hex(0, -1, false), "0x09");

        // Change type
        val.set_i64(0, 255 as i64);
        assert_eq!(val.hex(0, -1, false), "0xff 00 00 00 00 00 00 00");
        assert_eq!(val.kind().to_string(), TypesKind::I64.to_string());
        assert_eq!(val.ty, Types::i64());

        // Only Write in data, don't change type
        val.set_8bit(0, 64 as u8);
        assert_eq!(val.hex(0, -1, false), "0x40 00 00 00 00 00 00 00");
        assert_eq!(val.to_string(), "64");

        // Check binary
        let mut val = Value::from_string("1296");
        assert_eq!(val.get_i32(0), 1296 as i32);
        assert_eq!(val.bin(0, -1, false), "0b00010000 00000101 00000000 00000000");
        val.set_bit(0, 3, 18, "1");
        assert_eq!(val.bin(0, -1, false), "0b11111000 11111111 00000111 00000000");
        val.set_bit(1, 0, 7, "~");
        assert_eq!(val.bin(0, -1, false), "0b11111000 00000000 00000111 00000000");
        val.set_byte(2, 9, "^");    // need to enhance this
        assert_eq!(val.bin(0, -1, false), "0b11111000 00000000 00001110 00000000");

        // `set_type` by array
        val.set_type(Types::array(Types::u32(), 3));
        assert_eq!(val.kind().to_string(), "[u32; 3]");
        assert_eq!(val.size(), 12);

        // `set_type` by tuple
        val.set_type(Types::tuple(vec![Types::u32(), Types::u64()]));
        assert_eq!(val.kind().to_string(), "(u32, u64)");
        assert_eq!(val.size(), 12);

    }

    #[test]
    fn val_name() {
        let val = Value::from_string("(53, 31) : (i32, i64)");
        println!("{} : {}, ptr: {}, size: {}, val: {}", val.name(), val.ty.kind(), val.is_type_ptr(), val.size(), val.get_i32(0));
        let val = Value::from_string("&0x2e 35 : **(i32, i32)");
        println!("{} : {}, ptr: {}, size: {}, val: {}, ref_val: {}, ref_ref_val: {}", val.name(), val.ty.kind(), val.is_type_ptr(), val.size(), val.get_u32(0), val.get_ptr_ref(), val.get_ptr_ref().get_ptr_ref());
    }

    #[test]
    fn val_from() {

        let val = Value::from_string("23.5");
        assert_eq!(val.get_f32(0), 23.5f32);

        let val = Value::from_string("[18, 22, 623]");
        assert_eq!(val.get_i32(0), 18);
        assert_eq!(val.get_i32(4), 22);
        assert_eq!(val.get_i32(8), 623);

        let val = Value::from_string("[-18, 1, -23]");
        assert_eq!(val.get_i32(0), -18);
        assert_eq!(val.get_i32(4), 1);
        assert_eq!(val.get_i32(8), -23);

        let val = Value::from_string("[-18.5, 1.5, -23.5]");
        assert_eq!(val.get_f32(0), -18.5f32);
        assert_eq!(val.get_f32(4), 1.5f32);
        assert_eq!(val.get_f32(8), -23.5f32);

        let val = Value::from_string("(-18.5, 0, -23.5, \"hello\")");
        assert_eq!(val.get_f32(0), -18.5f32);
        assert_eq!(val.get_i32(4), 0);
        assert_eq!(val.get_f32(8), -23.5f32);
        assert_eq!(val.get_str(13), "ello");

        let mut val = Value::from_string("\"hallo\"");
        val.set_char(1, 'e');
        assert_eq!(val.get_char(1), 'e');
        assert_eq!(val.get_str(1), "ello");

        let val = Value::from_string("0b00001010 11001010 1111");
        assert_eq!(val.bin(0, -1, false), "0b00001010 11001010 11110000");
        let mut val1 = Value::from_string("0b00010000 00000101 00000000 00000000");
        let val2 = Value::from_string("0B00010000 00000101 00000000 00000000");
        assert_eq!(val1.bin(0, -1, true).replace("0B", ""), val2.bin(0, -1, false).replace("0b", ""));
        assert_eq!(val1.bin(0, -1, false).replace("0b", ""), val2.bin(0, -1, true).replace("0B", ""));

        assert_eq!(val1.bin(0, -1, false), "0b00010000 00000101 00000000 00000000");
        val1.set_bits(1, "10..1.10 ...111.0", false, false);
        assert_eq!(val1.bin(0, -1, false), "0b00010000 10001110 00011100 00000000");

        let mut val = Value::from_string("0x12345678");
        assert_eq!(val.hex(0, -1, false), "0x12 34 56 78");
        val.set_byte(0, 32, "c");
        assert_eq!(val.hex(0, -1, false), "0x20 34 56 78");

        let val = Value::from_string("0XC40CFF00");
        assert_eq!(val.hex(0, -1, true), "0XC4 0C FF 00");
        
    }


    #[test]
    fn val_change() {
        let mut val = Value::new(Types::u8());
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


    #[test]
    fn val_bits() {

        let mut val = Value::u32(0);
        val.set_ubyte(0, 0, 5, 3);
        assert_eq!(val.bin(0, -1, false), "0b00000011 00000000 00000000 00000000");

        let mut val = Value::u28(0);
        val.set_ubyte(0, 2, 8, 255);
        assert_eq!(val.bin_scale(0, -1, true), "0B0000 00000000 00000011 11111100");

        let mut val = Value::u7(127);
        val.set_ubyte(0, 2, 3, 5);
        assert_eq!(val.bin(0, -1, false), "0b01110111");
        val.set_ubyte(0, 1, 6, 0);
        assert_eq!(val.bin(0, -1, false), "0b00000001");
        val.set_ubyte(0, 2, 5, 31);
        assert_eq!(val.bin(0, -1, false), "0b01111101");

        let mut val = Value::u32(0);
        val.set_uhalf(0, 15, 16, 8253);
        assert_eq!(val.bin(0, -1, true), "0B00010000 00011110 10000000 00000000");
        assert_eq!(val.get_uhalf(0, 15, 16), 8253);
        val.set_uhalf(0, 22, 10, 1023);  
        assert_eq!(val.bin(0, -1, true), "0B11111111 11011110 10000000 00000000");

        let mut val = Value::u16(0);
        val.set_uhalf(0, 7, 9, 1023);
        assert_eq!(val.bin(0, -1, true), "0B11111111 10000000");

        let mut val = Value::u64(0);
        val.set_uword(0, 19, 31, 0x0FFFFFFF);
        assert_eq!(val.bin(0, -1, true), "0B00000000 00000000 01111111 11111111 11111111 11111000 00000000 00000000");
        assert_eq!(val.get_uword(0, 19, 31), 0x0FFFFFFF);

        // But a 32 maybe when scale > 8 and offset > 8

        let val = Value::u12(232);
        assert_eq!(val.hex(0, -1, false), "0xe8 00");


        let mut val = Value::ubit(19, 0x5CFCF);
        assert_eq!(val.bin_scale(0, -1, false), "0b11001111 11001111 101");
        let val2 = Value::bit(21, 0x1AF0FF);
        assert_eq!(val2.bin_scale(0, -1, false), "0b11111111 11110000 11010");
        val.concat(val2);
        assert_eq!(val.bin_scale(0, -1, false), "0b11001111 11001111 11111101 10000111 11010111");

        let mut val1 = Value::bit(12, 0xFFFFFFF);
        let val2 = Value::bit(31, 0x0);
        let val3 = Value::bit(29, 0xAAAAAAA);
        let val4 = Value::bit(66, 0xFFFFFFFF);
        val1.concat(val2).concat(val3).concat(val4);
        assert_eq!("b'138", val1.ty.to_string());
    }

}