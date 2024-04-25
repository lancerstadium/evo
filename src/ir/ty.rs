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

    // Unsigned bits
    U1, U2, U3, U4, U5, U6, U7,

    // Additional Unsigned bits
    U9, U10, U11, U12, U13, U14, U15,

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
            IRTypeKind::U1 => "u1".to_string(),
            IRTypeKind::U2 => "u2".to_string(),
            IRTypeKind::U3 => "u3".to_string(),
            IRTypeKind::U4 => "u4".to_string(),
            IRTypeKind::U5 => "u5".to_string(),
            IRTypeKind::U6 => "u6".to_string(),
            IRTypeKind::U7 => "u7".to_string(),
            IRTypeKind::U8 => "u8".to_string(),
            IRTypeKind::U9 => "u9".to_string(),
            IRTypeKind::U10 => "u10".to_string(),
            IRTypeKind::U11 => "u11".to_string(),
            IRTypeKind::U12 => "u12".to_string(),
            IRTypeKind::U13 => "u13".to_string(),
            IRTypeKind::U14 => "u14".to_string(),
            IRTypeKind::U15 => "u15".to_string(),
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
            "u1" => IRTypeKind::U1,
            "u2" => IRTypeKind::U2,
            "u3" => IRTypeKind::Array(IRType::u1(), 3),
            "u4" => IRTypeKind::Array(IRType::u1(), 4),
            "u5" => IRTypeKind::Array(IRType::u1(), 5),
            "u6" => IRTypeKind::Array(IRType::u1(), 6),
            "u7" => IRTypeKind::Array(IRType::u1(), 7),
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


    // ==================== IRType.is ====================== //

    /// TypeKind is signed
    pub fn is_signed(&self) -> bool {
        match self {
            IRTypeKind::I8
            | IRTypeKind::I16
            | IRTypeKind::I32
            | IRTypeKind::I64
            | IRTypeKind::I128
            | IRTypeKind::F16
            | IRTypeKind::F32
            | IRTypeKind::F64
            | IRTypeKind::F128 => true,
            _ => false
        }
    }

    /// TypeKind is float
    pub fn is_float(&self) -> bool {
        match self {
            IRTypeKind::F16
            | IRTypeKind::F32
            | IRTypeKind::F64
            | IRTypeKind::F128 => true,
            _ => false
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
        static TYPE_POOL: RefCell<HashMap<IRTypeKind, IRType>> = RefCell::new(HashMap::new());
        /// Size of pointers.
        static PTR_SIZE: Cell<usize> = Cell::new(mem::size_of::<*const ()>());
    }

    // ==================== IRType.get ===================== //

    /// Returns a type by the given `IRTypeKind`.
    pub fn get(type_kind: IRTypeKind) -> IRType {
        Self::TYPE_POOL.with(|pool| {
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

    /// Returns an `u1` type.
    pub fn u1() -> IRType {
        IRType::get(IRTypeKind::U1)
    }

    /// Returns an `u2` type.
    pub fn u2() -> IRType {
        IRType::get(IRTypeKind::U2)
    }

    /// Returns an `u3` type.
    pub fn u3() -> IRType {
        IRType::get(IRTypeKind::U3)
    }

    /// Returns an `u4` type.
    pub fn u4() -> IRType {
        IRType::get(IRTypeKind::U4)
    }

    /// Returns an `u5` type.
    pub fn u5() -> IRType {
        IRType::get(IRTypeKind::U5)
    }

    /// Returns an `u6` type.
    pub fn u6() -> IRType {
        IRType::get(IRTypeKind::U6)
    }

    /// Returns an `u7` type.
    pub fn u7() -> IRType {
        IRType::get(IRTypeKind::U7)
    }

    /// Returns an `u8` type.
    pub fn u8() -> IRType {
        IRType::get(IRTypeKind::U8)
    }

    /// Returns an `u9` type.
    pub fn u9() -> IRType {
        IRType::get(IRTypeKind::U9)
    }

    /// Returns an `u10` type.
    pub fn u10() -> IRType {
        IRType::get(IRTypeKind::U10)
    }

    /// Returns an `u11` type.
    pub fn u11() -> IRType {
        IRType::get(IRTypeKind::U11)
    }

    /// Returns an `u12` type.
    pub fn u12() -> IRType {
        IRType::get(IRTypeKind::U12)
    }

    /// Returns an `u13` type.
    pub fn u13() -> IRType {
        IRType::get(IRTypeKind::U13)
    }

    /// Returns an `u14` type.
    pub fn u14() -> IRType {
        IRType::get(IRTypeKind::U14)
    }

    /// Returns an `u15` type.
    pub fn u15() -> IRType {
        IRType::get(IRTypeKind::U15)
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
            IRTypeKind::U1 => 1,
            IRTypeKind::U2 => 1,
            IRTypeKind::U3 => 1,
            IRTypeKind::U4 => 1,
            IRTypeKind::U5 => 1,
            IRTypeKind::U6 => 1,
            IRTypeKind::U7 => 1,
            IRTypeKind::U8 => 1,
            IRTypeKind::U9 => 2,
            IRTypeKind::U10 => 2,
            IRTypeKind::U11 => 2,
            IRTypeKind::U12 => 2,
            IRTypeKind::U13 => 2,
            IRTypeKind::U14 => 2,
            IRTypeKind::U15 => 2,
            IRTypeKind::U16 => 2,
            IRTypeKind::U32 => 4,
            IRTypeKind::U64 => 8,
            IRTypeKind::U128 => 16,
            IRTypeKind::F16 => 2,
            IRTypeKind::F32 => 4,
            IRTypeKind::F64 => 8,
            IRTypeKind::F128 => 16,
            IRTypeKind::Array(ty, len) => ty.size() * len,
            IRTypeKind::Tuple(tys) => tys.iter().map(|ty| ty.size()).sum(),
            IRTypeKind::Ptr(..) | IRTypeKind::Func(..) | IRTypeKind::Struct(..) => Self::PTR_SIZE.with(|ptr_size| ptr_size.get()),
        }
    }

    /// Returns the scale vec of current type: bits number
    pub fn scale(&self) -> Vec<usize> {
        match self.kind() {
            IRTypeKind::U1 => vec![1],
            IRTypeKind::U2 => vec![2],
            IRTypeKind::U3 => vec![3],
            IRTypeKind::U4 => vec![4],
            IRTypeKind::U5 => vec![5],
            IRTypeKind::U6 => vec![6],
            IRTypeKind::U7 => vec![7],
            IRTypeKind::I8 | IRTypeKind::U8 => vec![8], 
            IRTypeKind::U9 => vec![9],
            IRTypeKind::U10 => vec![10],
            IRTypeKind::U11 => vec![11],
            IRTypeKind::U12 => vec![12],
            IRTypeKind::U13 => vec![13],
            IRTypeKind::U14 => vec![14],
            IRTypeKind::U15 => vec![15],
            IRTypeKind::I16 | IRTypeKind::U16 | IRTypeKind::F16 => vec![16],
            IRTypeKind::I32 | IRTypeKind::U32 | IRTypeKind::F32 => vec![32],
            IRTypeKind::I64 | IRTypeKind::U64 | IRTypeKind::F64 => vec![64],
            IRTypeKind::I128 | IRTypeKind::U128 | IRTypeKind::F128 => vec![128],
            // [u32; 5] => scale = [32, 32, 32, 32, 32]
            IRTypeKind::Array(ty, len) => (0..*len).map(|_| ty.size() * 8).collect(),
            // ((u32, u32), u32) => scale = [64, 32]
            IRTypeKind::Tuple(tys) => tys.iter().map(|ty| ty.size() * 8).collect(),
            IRTypeKind::Ptr(..) | IRTypeKind::Func(..) | IRTypeKind::Struct(..) => Self::PTR_SIZE.with(|ptr_size| vec![ptr_size.get() * 8]),
        }
    }

    /// Return types vec of current type
    pub fn types(&self) -> Vec<IRType> {
        match self.kind() {
            IRTypeKind::U1 | IRTypeKind::U2 | IRTypeKind::U3 | IRTypeKind::U4 | IRTypeKind::U5 | IRTypeKind::U6 | IRTypeKind::U7
                | IRTypeKind::U9 | IRTypeKind::U10 | IRTypeKind::U11 | IRTypeKind::U12 | IRTypeKind::U13 | IRTypeKind::U14 | IRTypeKind::U15
                | IRTypeKind:: I8 | IRTypeKind::I16 | IRTypeKind::I32 | IRTypeKind::I64 | IRTypeKind::I128
                | IRTypeKind::U8 | IRTypeKind::U16 | IRTypeKind::U32 | IRTypeKind::U64 | IRTypeKind::U128
                | IRTypeKind::F16 | IRTypeKind::F32 | IRTypeKind::F64 | IRTypeKind::F128 => vec![self.clone()],
            IRTypeKind::Ptr(ty) => vec![ty.clone()],
            IRTypeKind::Func(args, ret) => args.clone().into_iter().chain(vec![ret.clone()]).collect(),
            IRTypeKind::Array(ty, len) => (0..*len).map(|_| ty.clone()).collect(), 
            IRTypeKind::Tuple(tys) => tys.clone(),
            IRTypeKind::Struct(fields) => fields.clone(),
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
        assert_eq!(IRType::from_string("u1"), IRType::u1());
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
        assert_eq!(format!("{}", IRType::u1()), "u1");
        assert_eq!(format!("{}", IRType::u2()), "u2");
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
        assert_eq!(IRType::u1().size(), 1);
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
        assert_eq!(IRType::tuple(vec![IRType::i32(), IRType::f32()]).size(), 8);
        assert_eq!(IRType::func(vec![IRType::i32(), IRType::f64()], IRType::f64()).size(), 4);
        assert_eq!(IRType::stc(vec![IRType::i32(), IRType::f32()]).size(), 4);
    } 


    #[test]
    fn type_scale() {
        assert_eq!(IRType::i32().scale(), vec![32]);
        assert_eq!(IRType::u1().scale(), vec![1]);
        assert_eq!(IRType::u7().scale(), vec![7]);

        assert_eq!(IRType::array(IRType::i32(), 5).scale(), vec![32, 32, 32, 32, 32]);
        assert_eq!(IRType::array(IRType::array(IRType::i32(), 5), 3).scale(), vec![160, 160, 160]);
        assert_eq!(IRType::tuple(vec![IRType::i32(), IRType::f32()]).scale(), vec![32, 32]);
        assert_eq!(IRType::stc(vec![IRType::i32(), IRType::f32()]).scale(), vec![64]);
        
        IRType::set_ptr_size(4);
        assert_eq!(IRType::ptr(IRType::f64()).scale(), vec![32]);
        assert_eq!(IRType::array(IRType::ptr(IRType::i32()), 5).scale(), vec![32, 32, 32, 32, 32]);
        assert_eq!(IRType::func(vec![IRType::i32(), IRType::f64()], IRType::f64()).scale(), vec![32]);
    }


    #[test]
    fn type_types() {
        assert_eq!(IRType::i32().types(), vec![IRType::i32()]);
        assert_eq!(IRType::array(IRType::i32(), 5).types(), vec![IRType::i32(), IRType::i32(), IRType::i32(), IRType::i32(), IRType::i32()]);
        assert_eq!(IRType::array(IRType::array(IRType::i32(), 5), 3).types(), vec![IRType::array(IRType::i32(), 5), IRType::array(IRType::i32(), 5), IRType::array(IRType::i32(), 5)]);
        assert_eq!(IRType::tuple(vec![IRType::i32(), IRType::f32()]).types(), vec![IRType::i32(), IRType::f32()]);
        assert_eq!(IRType::stc(vec![IRType::i32(), IRType::f32()]).types(), vec![IRType::i32(), IRType::f32()]);
        assert_eq!(IRType::func(vec![IRType::i32(), IRType::f64()], IRType::f64()).types(), vec![IRType::i32(), IRType::f64(), IRType::f64()]);
    }

}