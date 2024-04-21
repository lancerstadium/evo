//! `evo::ir::types`: Types' definition in the IR
//! 
//! ## Description
//! Every `evo-ir` value and function should have a type.
//! A type can be any of the following:
//! 1. `IntType`: integer type such as `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`.
//! 2. `FloatType`: floating point type such as `f16`, `f32`, `f64`, `f128`.
//! 3. `ArrayType`: array type, a list of values of the same type.
//! 4. `PtrType`: pointer type stored as a value's address.
//! 5. `TupleType`: tuple type, a list of values of different types.
//! 6. `FuncType`: function type, a list of argument types and return type.
//! 7. `StructType`: struct type, a list of field types.


// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::{cmp, fmt, hash, mem};




// ============================================================================== //
//                                 ty::IRTypeKind
// ============================================================================== //

/// `IRTypeKind`: evo-ir enum types
#[derive(Hash, Clone, PartialEq, Eq)]
pub enum IRTypeKind {
    // Integer type
    I8, I16, I32, I64, I128, U8, U16, U32, U64, U128,

    // Float type
    F16, F32, F64, F128,

    // Array type
    Array(IRType, usize),

    // Pointer type
    Ptr(IRType),

    // Tuple type
    Tuple(Vec<IRType>),

    // Function type
    Func(Vec<IRType>, IRType),

    // Struct type
    Struct(Vec<IRType>),

}

/// Get string for `IRTypeKind`.
impl IRTypeKind {

    /// Converts `IRTypeKind` to string.
    pub fn to_string(&self) -> String {
        match self {
            IRTypeKind::I8 => "i8".to_string(),
            IRTypeKind::I16 => "i16".to_string(),
            IRTypeKind::I32 => "i32".to_string(),
            IRTypeKind::I64 => "i64".to_string(),
            IRTypeKind::I128 => "i128".to_string(),
            IRTypeKind::U8 => "u8".to_string(),
            IRTypeKind::U16 => "u16".to_string(),
            IRTypeKind::U32 => "u32".to_string(),
            IRTypeKind::U64 => "u64".to_string(),
            IRTypeKind::U128 => "u128".to_string(),
            IRTypeKind::F16 => "f16".to_string(),
            IRTypeKind::F32 => "f32".to_string(),
            IRTypeKind::F64 => "f64".to_string(),
            IRTypeKind::F128 => "f128".to_string(),
            IRTypeKind::Array(ty, len) => format!("[{}; {}]", ty.to_string(), len),
            IRTypeKind::Tuple(tys) => format!("({})", tys.iter().map(|ty| ty.to_string()).collect::<Vec<String>>().join(", ")),
            IRTypeKind::Ptr(ty) => format!("*{}", ty.to_string()),
            IRTypeKind::Func(args, ret) => format!("({}) -> {}", args.iter().map(|ty| ty.to_string()).collect::<Vec<String>>().join(", "), ret.to_string()),
            IRTypeKind::Struct(fields) => format!("struct {{{}}}", fields.iter().map(|ty| ty.to_string()).collect::<Vec<String>>().join(", ")),
        }
    }

    /// Converts `string` to `IRTypeKind`.
    pub fn from_string(s: &str) -> IRTypeKind {
        // Array: [ty; len]
        if s.starts_with("[") && s.ends_with("]") {
            let s = &s[1..s.len()-1];
            // Every parts should deal with side space
            // find last `;`
            let pos = s.rfind(';').unwrap();
            let ty = IRType::from_string(s[..pos].trim());
            let len = s[pos+1..].trim().parse().unwrap();
            return IRTypeKind::Array(ty, len)
        }
        // Pointer: *ty
        if s.starts_with("*") {
            let s = &s[1..];
            let ty = IRType::from_string(s.trim());
            return IRTypeKind::Ptr(ty);
        }
        // Tuple: (ty1, ty2, ...)
        if s.starts_with("(") && s.ends_with(")") {
            let s = &s[1..s.len()-1];
            let parts = s.split(',');
            let tys = parts.map(|ty| IRType::from_string(ty.trim())).collect();
            return IRTypeKind::Tuple(tys);
        }
        // Function: (ty1, ty2, ...) -> ty
        if s.starts_with("(") {
            let s = &s[1..];
            let mut parts = s.split(") ->");
            let args = parts.next().unwrap().split(',').map(|ty| IRType::from_string(ty.trim())).collect();
            let ret = IRType::from_string(parts.next().unwrap().trim());
            return IRTypeKind::Func(args, ret);
        }
        // Struct: struct {ty1, ty2, ...}
        if s.starts_with("struct {") && s.ends_with("}") {
            let s = &s[8..s.len()-1];
            let parts = s.split(',');
            let tys = parts.map(|ty| IRType::from_string(ty.trim())).collect();
            return IRTypeKind::Struct(tys);
        }
        match s {
            "i8" => IRTypeKind::I8,
            "i16" => IRTypeKind::I16,
            "i32" => IRTypeKind::I32,
            "i64" => IRTypeKind::I64,
            "i128" => IRTypeKind::I128,
            "u8" => IRTypeKind::U8,
            "u16" => IRTypeKind::U16,
            "u32" => IRTypeKind::U32,
            "u64" => IRTypeKind::U64,
            "u128" => IRTypeKind::U128,
            "f16" => IRTypeKind::F16,
            "f32" => IRTypeKind::F32,
            "f64" => IRTypeKind::F64,
            "f128" => IRTypeKind::F128,
            _ => panic!("Invalid type: {}", s),
        }
    }
}

/// Set string for `IRTypeKind`.
impl fmt::Display for IRTypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


// ============================================================================== //
//                                    ty::IRType
// ============================================================================== //

/// `IRType`: evo-ir type
#[derive(Clone, Eq)]
pub struct IRType(Rc<IRTypeKind>);


impl IRType {

    // Init type pool.
    thread_local! {
        /// Pool of all created types.
        static POOL: RefCell<HashMap<IRTypeKind, IRType>> = RefCell::new(HashMap::new());
        /// Size of pointers.
        static PTR_SIZE: Cell<usize> = Cell::new(mem::size_of::<*const ()>());
    }

    // ==================== IRType.get ===================== //

    /// Returns a type by the given `IRTypeKind`.
    pub fn get(type_kind: IRTypeKind) -> IRType {
        Self::POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.get(&type_kind).cloned().unwrap_or_else(|| {
                let v = Self(Rc::new(type_kind.clone()));
                pool.insert(type_kind, v.clone());
                v
            })
        })
    }
    
    /// Returns an `i8` type.
    pub fn i8() -> IRType {
        IRType::get(IRTypeKind::I8)
    }

    /// Returns an `i16` type.
    pub fn i16() -> IRType {
        IRType::get(IRTypeKind::I16)
    }

    /// Returns an `i32` type.
    pub fn i32() -> IRType {
        IRType::get(IRTypeKind::I32)
    }

    /// Returns an `i64` type.
    pub fn i64() -> IRType {
        IRType::get(IRTypeKind::I64)
    }

    /// Returns an `i128` type.
    pub fn i128() -> IRType {
        IRType::get(IRTypeKind::I128)
    }

    /// Returns an `u8` type.
    pub fn u8() -> IRType {
        IRType::get(IRTypeKind::U8)
    }

    /// Returns an `u16` type.
    pub fn u16() -> IRType {
        IRType::get(IRTypeKind::U16)
    }

    /// Returns an `u32` type.
    pub fn u32() -> IRType {
        IRType::get(IRTypeKind::U32)
    }

    /// Returns an `u64` type.
    pub fn u64() -> IRType {
        IRType::get(IRTypeKind::U64)
    }

    /// Returns an `u128` type.
    pub fn u128() -> IRType {
        IRType::get(IRTypeKind::U128)
    }

    /// Returns an `f16` type.
    pub fn f16() -> IRType {
        IRType::get(IRTypeKind::F16)
    }

    /// Returns an `f32` type.
    pub fn f32() -> IRType {
        IRType::get(IRTypeKind::F32)
    }

    /// Returns an `f64` type.
    pub fn f64() -> IRType {
        IRType::get(IRTypeKind::F64)
    }

    /// Returns an `f128` type.
    pub fn f128() -> IRType {
        IRType::get(IRTypeKind::F128)
    }

    /// Returns an `array` type.
    pub fn array(ty: IRType, size: usize) -> IRType {
        assert!(size != 0, "array size cannot be 0");
        IRType::get(IRTypeKind::Array(ty, size))
    }

    /// Returns an `ptr` type.
    pub fn ptr(ty: IRType) -> IRType {
        IRType::get(IRTypeKind::Ptr(ty))
    }

    /// Returns an `func` type.
    pub fn func(args: Vec<IRType>, ret: IRType) -> IRType {
        IRType::get(IRTypeKind::Func(args, ret))
    }

    /// Returns an `tuple` type.
    pub fn tuple(tys: Vec<IRType>) -> IRType {
        IRType::get(IRTypeKind::Tuple(tys))
    }

    /// Returns an `struct` type.
    pub fn stc(fields: Vec<IRType>) -> IRType {
        IRType::get(IRTypeKind::Struct(fields))
    }

    /// Return a reference to the current type.
    pub fn kind(&self) -> &IRTypeKind {
        &self.0
    }

    /// Return the size of current type in bytes.
    pub fn size(&self) -> usize {
        match self.kind() {
            IRTypeKind::I8 => 1,
            IRTypeKind::I16 => 2,
            IRTypeKind::I32 => 4,
            IRTypeKind::I64 => 8,
            IRTypeKind::I128 => 16,
            IRTypeKind::U8 => 1,
            IRTypeKind::U16 => 2,
            IRTypeKind::U32 => 4,
            IRTypeKind::U64 => 8,
            IRTypeKind::U128 => 16,
            IRTypeKind::F16 => 2,
            IRTypeKind::F32 => 4,
            IRTypeKind::F64 => 8,
            IRTypeKind::F128 => 16,
            IRTypeKind::Array(ty, len) => ty.size() * len,
            IRTypeKind::Ptr(..) | IRTypeKind::Tuple(..) | IRTypeKind::Func(..) | IRTypeKind::Struct(..) => Self::PTR_SIZE.with(|ptr_size| ptr_size.get()),
        }
    }

    /// Returns a new `IRType` from string
    pub fn from_string(s: &str) -> IRType {
        IRType::get(IRTypeKind::from_string(s))
    }

    // ==================== IRType.set ===================== //

    /// Set IRType by Given `IRTypeKind`
    pub fn set(&mut self, kind: IRTypeKind) {
        self.0 = Rc::new(kind);
    }

    /// Sets the size of pointers.
    pub fn set_ptr_size(size: usize) {
        Self::PTR_SIZE.with(|ptr_size| {
            ptr_size.set(size);
        });
    }


}


impl cmp::PartialEq for IRType {
    /// Compare two `IRType`
    fn eq(&self, other: &Self) -> bool {
        // 1. Compare size
        if self.size() != other.size() {
            return false
        }
        // 2. Compare kind
        if self.kind() != other.kind() {
            return false
        }
        true
    }

    // fn eq(&self, other: &IRType) -> bool {
    //     Rc::ptr_eq(&self.0, &other.0)
    // }
}

impl fmt::Display for IRType {
    /// Format `IRType`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for IRType {
    /// Format `IRType`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl hash::Hash for IRType {
    /// Hash `IRType`
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}







// ============================================================================== //
//                                 Unit Tests
// ============================================================================== //


#[cfg(test)]
mod ty_test {

    use super::*;


    #[test]
    fn type_from_string() {
        assert_eq!(IRType::from_string("i8"), IRType::i8());
        assert_eq!(IRType::from_string("i16"), IRType::i16());
        assert_eq!(IRType::from_string("i32"), IRType::i32());
        assert_eq!(IRType::from_string("i64"), IRType::i64());
        assert_eq!(IRType::from_string("i128"), IRType::i128());
        assert_eq!(IRType::from_string("u8"), IRType::u8());
        assert_eq!(IRType::from_string("u16"), IRType::u16());
        assert_eq!(IRType::from_string("u32"), IRType::u32());
        assert_eq!(IRType::from_string("u64"), IRType::u64());
        assert_eq!(IRType::from_string("u128"), IRType::u128());
        assert_eq!(IRType::from_string("f16"), IRType::f16());
        assert_eq!(IRType::from_string("f32"), IRType::f32());
        assert_eq!(IRType::from_string("f64"), IRType::f64());

        assert_eq!(IRType::from_string("[i32; 10]"), IRType::array(IRType::i32(), 10));
        assert_eq!(IRType::from_string("[[i32; 10]; 3]"), IRType::array(IRType::array(IRType::i32(), 10), 3));
        assert_eq!(IRType::from_string("*i32"), IRType::ptr(IRType::i32()));

        assert_eq!(IRType::from_string("(i32, f64)"), IRType::tuple(vec![IRType::i32(), IRType::f64()]));
        assert_eq!(IRType::from_string("(i32, f64) -> f64"), IRType::func(vec![IRType::i32(), IRType::f64()], IRType::f64()));
        assert_eq!(IRType::from_string("struct {i32, f64}"), IRType::stc(vec![IRType::i32(), IRType::f64()]));
    }

    #[test]
    fn type_to_string() {
        assert_eq!(format!("{}", IRType::i8()), "i8");
        assert_eq!(format!("{}", IRType::i16()), "i16");
        assert_eq!(format!("{}", IRType::i32()), "i32");
        assert_eq!(format!("{}", IRType::i64()), "i64");
        assert_eq!(format!("{}", IRType::i128()), "i128");
        assert_eq!(format!("{}", IRType::u8()), "u8");
        assert_eq!(format!("{}", IRType::u16()), "u16");
        assert_eq!(format!("{}", IRType::u32()), "u32");
        assert_eq!(format!("{}", IRType::u64()), "u64");
        assert_eq!(format!("{}", IRType::u128()), "u128");
        assert_eq!(format!("{}", IRType::f16()), "f16");
        assert_eq!(format!("{}", IRType::f32()), "f32");
        assert_eq!(format!("{}", IRType::f64()), "f64");
        assert_eq!(format!("{}", IRType::f128()), "f128");
        assert_eq!(format!("{}", IRType::array(IRType::i32(), 10)), "[i32; 10]");
        assert_eq!(format!("{}", IRType::array(IRType::array(IRType::i32(), 10), 3)), "[[i32; 10]; 3]");
        assert_eq!(format!("{}", IRType::ptr(IRType::ptr(IRType::f64()))), "**f64");
        assert_eq!(format!("{}", IRType::func(vec![IRType::i32(), IRType::f64()], IRType::f64())), "(i32, f64) -> f64");
        assert_eq!(format!("{}", IRType::tuple(vec![IRType::i32(), IRType::f64()])), "(i32, f64)");
        assert_eq!(format!("{}", IRType::stc(vec![IRType::i32(), IRType::f64()])), "struct {i32, f64}");
    }

    #[test]
    fn type_eq() {
        assert_eq!(IRType::i8(), IRType::i8());
        assert_eq!(IRType::array(IRType::i32(), 6), IRType::array(IRType::i32(), 6));
    }

    #[test]
    fn type_size() {

        assert_eq!(IRType::i8().size(), 1);
        assert_eq!(IRType::i16().size(), 2);
        assert_eq!(IRType::i32().size(), 4);
        assert_eq!(IRType::i64().size(), 8);
        assert_eq!(IRType::i128().size(), 16);
        assert_eq!(IRType::u8().size(), 1);
        assert_eq!(IRType::u16().size(), 2);
        assert_eq!(IRType::u32().size(), 4);
        assert_eq!(IRType::u64().size(), 8);
        assert_eq!(IRType::u128().size(), 16);
        assert_eq!(IRType::f16().size(), 2);
        assert_eq!(IRType::f32().size(), 4);
        assert_eq!(IRType::f64().size(), 8);
        assert_eq!(IRType::f128().size(), 16);

        assert_eq!(IRType::array(IRType::i32(), 10).size(), 10 * 4);
        assert_eq!(IRType::array(IRType::array(IRType::i32(), 10), 3).size(), 3 * 10 * 4);
        assert_eq!(IRType::ptr(IRType::f64()).size(), mem::size_of::<usize>());

        IRType::set_ptr_size(4);
        assert_eq!(IRType::ptr(IRType::f64()).size(), 4);
        assert_eq!(IRType::array(IRType::ptr(IRType::i32()), 5).size(), 4 * 5);
        assert_eq!(IRType::func(vec![IRType::i32(), IRType::f64()], IRType::f64()).size(), 4);
        assert_eq!(IRType::tuple(vec![IRType::i32(), IRType::f32()]).size(), 4);
        assert_eq!(IRType::stc(vec![IRType::i32(), IRType::f32()]).size(), 4);
    } 

}