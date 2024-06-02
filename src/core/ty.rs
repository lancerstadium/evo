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
//                                 ty::TypesKind
// ============================================================================== //

/// `TypesKind`: evo-ir enum types
#[derive(Hash, Clone, PartialEq, Eq)]
pub enum TypesKind {
    // Void type
    Void,

    // Bit type
    Bit(usize),

    // Unsigned bits
    U1, U2, U3, U4, U5, U6, U7,

    // Additional Unsigned bits
    U9, U10, U11, U12, U13, U14, U15,

    // Additional Unsigned bits plus
    U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31,

    // Integer type
    I8, I16, I32, I64, I128, U8, U16, U32, U64, U128,

    // Float type
    F16, F32, F64, F128,

    // Array type
    Array(Types, usize),

    // Pointer type
    Ptr(Types),

    // Tuple type
    Tuple(Vec<Types>),

    // Function type
    Func(Vec<Types>, Types),

    // Struct type
    Struct(Vec<Types>),

    None
}

/// Get string for `TypesKind`.
impl TypesKind {

    /// Converts `TypesKind` to string.
    pub fn to_string(&self) -> String {
        match self {
            TypesKind::Void => "void".to_string(),
            TypesKind::None => "none".to_string(),
            TypesKind::Bit(n) => format!("b'{}", n),
            TypesKind::I8 => "i8".to_string(),
            TypesKind::I16 => "i16".to_string(),
            TypesKind::I32 => "i32".to_string(),
            TypesKind::I64 => "i64".to_string(),
            TypesKind::I128 => "i128".to_string(),
            TypesKind::U1 => "u1".to_string(),
            TypesKind::U2 => "u2".to_string(),
            TypesKind::U3 => "u3".to_string(),
            TypesKind::U4 => "u4".to_string(),
            TypesKind::U5 => "u5".to_string(),
            TypesKind::U6 => "u6".to_string(),
            TypesKind::U7 => "u7".to_string(),
            TypesKind::U8 => "u8".to_string(),
            TypesKind::U9 => "u9".to_string(),
            TypesKind::U10 => "u10".to_string(),
            TypesKind::U11 => "u11".to_string(),
            TypesKind::U12 => "u12".to_string(),
            TypesKind::U13 => "u13".to_string(),
            TypesKind::U14 => "u14".to_string(),
            TypesKind::U15 => "u15".to_string(),
            TypesKind::U16 => "u16".to_string(),
            TypesKind::U17 => "u17".to_string(),
            TypesKind::U18 => "u18".to_string(),
            TypesKind::U19 => "u19".to_string(),
            TypesKind::U20 => "u20".to_string(),
            TypesKind::U21 => "u21".to_string(),
            TypesKind::U22 => "u22".to_string(),
            TypesKind::U23 => "u23".to_string(),
            TypesKind::U24 => "u24".to_string(),
            TypesKind::U25 => "u25".to_string(),
            TypesKind::U26 => "u26".to_string(),
            TypesKind::U27 => "u27".to_string(),
            TypesKind::U28 => "u28".to_string(),
            TypesKind::U29 => "u29".to_string(),
            TypesKind::U30 => "u30".to_string(),
            TypesKind::U31 => "u31".to_string(),
            TypesKind::U32 => "u32".to_string(),
            TypesKind::U64 => "u64".to_string(),
            TypesKind::U128 => "u128".to_string(),
            TypesKind::F16 => "f16".to_string(),
            TypesKind::F32 => "f32".to_string(),
            TypesKind::F64 => "f64".to_string(),
            TypesKind::F128 => "f128".to_string(),
            TypesKind::Array(ty, len) => format!("[{}; {}]", ty.to_string(), len),
            TypesKind::Tuple(tys) => format!("({})", tys.iter().map(|ty| ty.to_string()).collect::<Vec<String>>().join(", ")),
            TypesKind::Ptr(ty) => format!("*{}", ty.to_string()),
            TypesKind::Func(args, ret) => format!("fn ({}) -> {}", args.iter().map(|ty| ty.to_string()).collect::<Vec<String>>().join(", "), ret.to_string()),
            TypesKind::Struct(fields) => format!("struct {{{}}}", fields.iter().map(|ty| ty.to_string()).collect::<Vec<String>>().join(", ")),
        }
    }

    /// Converts `string` to `TypesKind`.
    pub fn from_string(s: &str) -> TypesKind {
        // Bit: b'23
        if s.starts_with("b'") {
            let s = &s[2..];
            let n = s.parse().unwrap();
            return TypesKind::Bit(n)
        }
        // Array: [ty; len]
        if s.starts_with("[") && s.ends_with("]") {
            let s = &s[1..s.len()-1];
            // Every parts should deal with side space
            // find last `;`
            let pos = s.rfind(';').unwrap();
            let ty = Types::from_string(s[..pos].trim());
            let len = s[pos+1..].trim().parse().unwrap();
            return TypesKind::Array(ty, len)
        }
        // Pointer: *ty
        if s.starts_with("*") {
            let s = &s[1..];
            let ty = Types::from_string(s.trim());
            return TypesKind::Ptr(ty);
        }
        // Tuple: (ty1, ty2, ...)
        if s.starts_with("(") && s.ends_with(")") {
            let s = &s[1..s.len()-1];
            let parts = s.split(',');
            let tys = parts.map(|ty| Types::from_string(ty.trim())).collect();
            return TypesKind::Tuple(tys);
        }
        // Function: fn (ty1, ty2, ...) -> ty
        if s.starts_with("fn (") {
            let s = &s[4..];
            let parts = s.split(") ->").map(|ty| ty.trim()).collect::<Vec<&str>>();
            let args;
            let ret;
            if parts.len() < 2 {
                args = Vec::new();
                ret = Types::from_string(parts[0].trim());
            } else {
                args = parts[0].split(',').map(|ty| Types::from_string(ty.trim())).collect();
                ret = Types::from_string(parts[1].trim());
            }
            return TypesKind::Func(args, ret);
        }
        // Struct: struct {ty1, ty2, ...}
        if s.starts_with("struct {") && s.ends_with("}") {
            let s = &s[8..s.len()-1];
            let parts = s.split(',');
            let tys = parts.map(|ty| Types::from_string(ty.trim())).collect();
            return TypesKind::Struct(tys);
        }
        match s {
            "void" => TypesKind::Void,
            "none" => TypesKind::None,
            "i8" => TypesKind::I8,
            "i16" => TypesKind::I16,
            "i32" => TypesKind::I32,
            "i64" => TypesKind::I64,
            "i128" => TypesKind::I128,
            "u1" => TypesKind::U1,
            "u2" => TypesKind::U2,
            "u3" => TypesKind::Array(Types::u1(), 3),
            "u4" => TypesKind::Array(Types::u1(), 4),
            "u5" => TypesKind::Array(Types::u1(), 5),
            "u6" => TypesKind::Array(Types::u1(), 6),
            "u7" => TypesKind::Array(Types::u1(), 7),
            "u8" => TypesKind::U8,
            "u9" => TypesKind::U9,
            "u10" => TypesKind::U10,
            "u11" => TypesKind::U11,
            "u12" => TypesKind::U12,
            "u13" => TypesKind::U13,
            "u14" => TypesKind::U14,
            "u15" => TypesKind::U15,
            "u16" => TypesKind::U16,
            "u17" => TypesKind::U17,
            "u18" => TypesKind::U18,
            "u19" => TypesKind::U19,
            "u20" => TypesKind::U20,
            "u21" => TypesKind::U21,
            "u22" => TypesKind::U22,
            "u23" => TypesKind::U23,
            "u24" => TypesKind::U24,
            "u25" => TypesKind::U25,
            "u26" => TypesKind::U26,
            "u27" => TypesKind::U27,
            "u28" => TypesKind::U28,
            "u29" => TypesKind::U29,
            "u30" => TypesKind::U30,
            "u31" => TypesKind::U31,
            "u32" => TypesKind::U32,
            "u64" => TypesKind::U64,
            "u128" => TypesKind::U128,
            "f16" => TypesKind::F16,
            "f32" => TypesKind::F32,
            "f64" => TypesKind::F64,
            "f128" => TypesKind::F128,
            _ => panic!("Invalid type: {}", s),
        }
    }


    // ==================== Types.is ====================== //

    /// TypeKind is signed
    pub fn is_signed(&self) -> bool {
        match self {
            TypesKind::I8
            | TypesKind::I16
            | TypesKind::I32
            | TypesKind::I64
            | TypesKind::I128
            | TypesKind::F16
            | TypesKind::F32
            | TypesKind::F64
            | TypesKind::F128 => true,
            _ => false
        }
    }

    /// TypeKind is float
    pub fn is_float(&self) -> bool {
        match self {
            TypesKind::F16
            | TypesKind::F32
            | TypesKind::F64
            | TypesKind::F128 => true,
            _ => false
        }
    }

    /// TypeKind is bit
    pub fn is_bit(&self) -> bool {
        match self {
            TypesKind::Bit(_) => true,
            _ => false
        }
    }

}

/// Set string for `TypesKind`.
impl fmt::Display for TypesKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


// ============================================================================== //
//                                    ty::Types
// ============================================================================== //

/// `Types`: evo-ir type
#[derive(Clone, Eq)]
pub struct Types(Rc<TypesKind>);


impl Types {

    // Init type pool.
    thread_local! {
        /// Pool of all created types.
        static TYPE_POOL: RefCell<HashMap<TypesKind, Types>> = RefCell::new(HashMap::new());
        /// Size of pointers.
        static PTR_SIZE: Cell<usize> = Cell::new(mem::size_of::<*const ()>());
    }

    // ==================== Types.get ===================== //

    /// Returns a type by the given `TypesKind`.
    pub fn get(type_kind: TypesKind) -> Types {
        Self::TYPE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.get(&type_kind).cloned().unwrap_or_else(|| {
                let v = Self(Rc::new(type_kind.clone()));
                pool.insert(type_kind, v.clone());
                v
            })
        })
    }
    
    /// Returns an `void` type.
    pub fn void() -> Types {
        Types::get(TypesKind::Void)
    }

    /// Returns an `none` type.
    pub fn none() -> Types {
        Types::get(TypesKind::None)
    }

    /// Returns an `bit` type: scale is the width of bit.
    pub fn bit(scale: usize) -> Types {
        Types::get(TypesKind::Bit(scale))
    }
    
    /// Returns an `i8` type.
    pub fn i8() -> Types {
        Types::get(TypesKind::I8)
    }

    /// Returns an `i16` type.
    pub fn i16() -> Types {
        Types::get(TypesKind::I16)
    }

    /// Returns an `i32` type.
    pub fn i32() -> Types {
        Types::get(TypesKind::I32)
    }

    /// Returns an `i64` type.
    pub fn i64() -> Types {
        Types::get(TypesKind::I64)
    }

    /// Returns an `i128` type.
    pub fn i128() -> Types {
        Types::get(TypesKind::I128)
    }

    /// Returns an `u1` type.
    pub fn u1() -> Types {
        Types::get(TypesKind::U1)
    }

    /// Returns an `u2` type.
    pub fn u2() -> Types {
        Types::get(TypesKind::U2)
    }

    /// Returns an `u3` type.
    pub fn u3() -> Types {
        Types::get(TypesKind::U3)
    }

    /// Returns an `u4` type.
    pub fn u4() -> Types {
        Types::get(TypesKind::U4)
    }

    /// Returns an `u5` type.
    pub fn u5() -> Types {
        Types::get(TypesKind::U5)
    }

    /// Returns an `u6` type.
    pub fn u6() -> Types {
        Types::get(TypesKind::U6)
    }

    /// Returns an `u7` type.
    pub fn u7() -> Types {
        Types::get(TypesKind::U7)
    }

    /// Returns an `u8` type.
    pub fn u8() -> Types {
        Types::get(TypesKind::U8)
    }

    /// Returns an `u9` type.
    pub fn u9() -> Types {
        Types::get(TypesKind::U9)
    }

    /// Returns an `u10` type.
    pub fn u10() -> Types {
        Types::get(TypesKind::U10)
    }

    /// Returns an `u11` type.
    pub fn u11() -> Types {
        Types::get(TypesKind::U11)
    }

    /// Returns an `u12` type.
    pub fn u12() -> Types {
        Types::get(TypesKind::U12)
    }

    /// Returns an `u13` type.
    pub fn u13() -> Types {
        Types::get(TypesKind::U13)
    }

    /// Returns an `u14` type.
    pub fn u14() -> Types {
        Types::get(TypesKind::U14)
    }

    /// Returns an `u15` type.
    pub fn u15() -> Types {
        Types::get(TypesKind::U15)
    }

    /// Returns an `u16` type.
    pub fn u16() -> Types {
        Types::get(TypesKind::U16)
    }

    /// Returns an `u17` type.
    pub fn u17() -> Types {
        Types::get(TypesKind::U17)
    }

    /// Returns an `u18` type.
    pub fn u18() -> Types {
        Types::get(TypesKind::U18)
    }

    /// Returns an `u19` type.
    pub fn u19() -> Types {
        Types::get(TypesKind::U19)
    }

    /// Returns an `u20` type.
    pub fn u20() -> Types {
        Types::get(TypesKind::U20)
    }

    /// Returns an `u21` type.
    pub fn u21() -> Types {
        Types::get(TypesKind::U21)
    }

    /// Returns an `u22` type.
    pub fn u22() -> Types {
        Types::get(TypesKind::U22)
    }

    /// Returns an `u23` type.
    pub fn u23() -> Types {
        Types::get(TypesKind::U23)
    }

    /// Returns an `u24` type.
    pub fn u24() -> Types {
        Types::get(TypesKind::U24)
    }

    /// Returns an `u25` type.
    pub fn u25() -> Types {
        Types::get(TypesKind::U25)
    }

    /// Returns an `u26` type.
    pub fn u26() -> Types {
        Types::get(TypesKind::U26)
    }

    /// Returns an `u27` type.
    pub fn u27() -> Types {
        Types::get(TypesKind::U27)
    }

    /// Returns an `u28` type.
    pub fn u28() -> Types {
        Types::get(TypesKind::U28)
    }

    /// Returns an `u29` type.
    pub fn u29() -> Types {
        Types::get(TypesKind::U29)
    }

    /// Returns an `u30` type.
    pub fn u30() -> Types {
        Types::get(TypesKind::U30)
    }

    /// Returns an `u31` type.
    pub fn u31() -> Types {
        Types::get(TypesKind::U31)
    }

    /// Returns an `u32` type.
    pub fn u32() -> Types {
        Types::get(TypesKind::U32)
    }

    /// Returns an `u64` type.
    pub fn u64() -> Types {
        Types::get(TypesKind::U64)
    }

    /// Returns an `u128` type.
    pub fn u128() -> Types {
        Types::get(TypesKind::U128)
    }

    /// Returns an `f16` type.
    pub fn f16() -> Types {
        Types::get(TypesKind::F16)
    }

    /// Returns an `f32` type.
    pub fn f32() -> Types {
        Types::get(TypesKind::F32)
    }

    /// Returns an `f64` type.
    pub fn f64() -> Types {
        Types::get(TypesKind::F64)
    }

    /// Returns an `f128` type.
    pub fn f128() -> Types {
        Types::get(TypesKind::F128)
    }

    /// Returns an `array` type.
    pub fn array(ty: Types, size: usize) -> Types {
        Types::get(TypesKind::Array(ty, size))
    }

    /// Returns an `ptr` type.
    pub fn ptr(ty: Types) -> Types {
        Types::get(TypesKind::Ptr(ty))
    }

    /// Returns an `func` type.
    pub fn func(args: Vec<Types>, ret: Types) -> Types {
        Types::get(TypesKind::Func(args, ret))
    }

    /// Returns an `tuple` type.
    pub fn tuple(tys: Vec<Types>) -> Types {
        Types::get(TypesKind::Tuple(tys))
    }

    /// Returns an `struct` type.
    pub fn stc(fields: Vec<Types>) -> Types {
        Types::get(TypesKind::Struct(fields))
    }

    /// Return a reference to the current type.
    pub fn kind(&self) -> &TypesKind {
        &self.0
    }

    /// Return the size of current type in bytes.
    pub fn size(&self) -> usize {
        match self.kind() {
            TypesKind::Void => 0,
            TypesKind::None => 0,
            TypesKind::Bit(n) => n / 8 + (n % 8 != 0) as usize,
            TypesKind::I8 => 1,
            TypesKind::I16 => 2,
            TypesKind::I32 => 4,
            TypesKind::I64 => 8,
            TypesKind::I128 => 16,
            TypesKind::U1 => 1,
            TypesKind::U2 => 1,
            TypesKind::U3 => 1,
            TypesKind::U4 => 1,
            TypesKind::U5 => 1,
            TypesKind::U6 => 1,
            TypesKind::U7 => 1,
            TypesKind::U8 => 1,
            TypesKind::U9 => 2,
            TypesKind::U10 => 2,
            TypesKind::U11 => 2,
            TypesKind::U12 => 2,
            TypesKind::U13 => 2,
            TypesKind::U14 => 2,
            TypesKind::U15 => 2,
            TypesKind::U16 => 2,
            TypesKind::U17 => 3,
            TypesKind::U18 => 3,
            TypesKind::U19 => 3,
            TypesKind::U20 => 3,
            TypesKind::U21 => 3,
            TypesKind::U22 => 3,
            TypesKind::U23 => 3,
            TypesKind::U24 => 3,
            TypesKind::U25 => 4,
            TypesKind::U26 => 4,
            TypesKind::U27 => 4,
            TypesKind::U28 => 4,
            TypesKind::U29 => 4,
            TypesKind::U30 => 4,
            TypesKind::U31 => 4,
            TypesKind::U32 => 4,
            TypesKind::U64 => 8,
            TypesKind::U128 => 16,
            TypesKind::F16 => 2,
            TypesKind::F32 => 4,
            TypesKind::F64 => 8,
            TypesKind::F128 => 16,
            TypesKind::Array(ty, len) => ty.size() * len,
            TypesKind::Tuple(tys) => tys.iter().map(|ty| ty.size()).sum(),
            TypesKind::Ptr(..) | TypesKind::Func(..) | TypesKind::Struct(..) => Self::PTR_SIZE.with(|ptr_size| ptr_size.get()),
        }
    }

    /// Returns the scale vec of current type: bits number
    pub fn scale(&self) -> Vec<usize> {
        match self.kind() {
            TypesKind::Void => vec![0],
            TypesKind::None => vec![0],
            TypesKind::Bit(n) => vec![*n],
            TypesKind::U1 => vec![1],
            TypesKind::U2 => vec![2],
            TypesKind::U3 => vec![3],
            TypesKind::U4 => vec![4],
            TypesKind::U5 => vec![5],
            TypesKind::U6 => vec![6],
            TypesKind::U7 => vec![7],
            TypesKind::I8 | TypesKind::U8 => vec![8], 
            TypesKind::U9 => vec![9],
            TypesKind::U10 => vec![10],
            TypesKind::U11 => vec![11],
            TypesKind::U12 => vec![12],
            TypesKind::U13 => vec![13],
            TypesKind::U14 => vec![14],
            TypesKind::U15 => vec![15],
            TypesKind::I16 | TypesKind::U16 | TypesKind::F16 => vec![16],
            TypesKind::U17 => vec![17],
            TypesKind::U18 => vec![18],
            TypesKind::U19 => vec![19],
            TypesKind::U20 => vec![20],
            TypesKind::U21 => vec![21],
            TypesKind::U22 => vec![22],
            TypesKind::U23 => vec![23],
            TypesKind::U24 => vec![24],
            TypesKind::U25 => vec![25],
            TypesKind::U26 => vec![26],
            TypesKind::U27 => vec![27],
            TypesKind::U28 => vec![28],
            TypesKind::U29 => vec![29],
            TypesKind::U30 => vec![30],
            TypesKind::U31 => vec![31],
            TypesKind::I32 | TypesKind::U32 | TypesKind::F32 => vec![32],
            TypesKind::I64 | TypesKind::U64 | TypesKind::F64 => vec![64],
            TypesKind::I128 | TypesKind::U128 | TypesKind::F128 => vec![128],
            // [u32; 5] => scale = [32, 32, 32, 32, 32]
            TypesKind::Array(ty, len) => (0..*len).map(|_| ty.size() * 8).collect(),
            // ((u32, u32), u32) => scale = [64, 32]
            TypesKind::Tuple(tys) => tys.iter().map(|ty| ty.size() * 8).collect(),
            TypesKind::Ptr(..) | TypesKind::Func(..) | TypesKind::Struct(..) => Self::PTR_SIZE.with(|ptr_size| vec![ptr_size.get() * 8]),
        }
    }

    /// Return types vec of current type
    pub fn types(&self) -> Vec<Types> {
        match self.kind() {
            TypesKind::Void | TypesKind::None | TypesKind::Bit(_) | TypesKind::U1 | TypesKind::U2 | TypesKind::U3 | TypesKind::U4 | TypesKind::U5 | TypesKind::U6 | TypesKind::U7
                | TypesKind::U9 | TypesKind::U10 | TypesKind::U11 | TypesKind::U12 | TypesKind::U13 | TypesKind::U14 | TypesKind::U15
                | TypesKind::U17 | TypesKind::U18 | TypesKind::U19 | TypesKind::U20 | TypesKind::U21 | TypesKind::U22 | TypesKind::U23
                | TypesKind::U24 | TypesKind::U25 | TypesKind::U26 | TypesKind::U27 | TypesKind::U28 | TypesKind::U29 | TypesKind::U30
                | TypesKind::U31
                | TypesKind:: I8 | TypesKind::I16 | TypesKind::I32 | TypesKind::I64 | TypesKind::I128
                | TypesKind::U8 | TypesKind::U16 | TypesKind::U32 | TypesKind::U64 | TypesKind::U128
                | TypesKind::F16 | TypesKind::F32 | TypesKind::F64 | TypesKind::F128 => vec![self.clone()],
            TypesKind::Ptr(ty) => vec![ty.clone()],
            TypesKind::Func(args, ret) => args.clone().into_iter().chain(vec![ret.clone()]).collect(),
            TypesKind::Array(ty, len) => (0..*len).map(|_| ty.clone()).collect(), 
            TypesKind::Tuple(tys) => tys.clone(),
            TypesKind::Struct(fields) => fields.clone(),
        }
    }


    /// Returns a new `Types` from string
    pub fn from_string(s: &str) -> Types {
        Types::get(TypesKind::from_string(s))
    }

    // ==================== Types.set ===================== //

    /// Set Types by Given `TypesKind`
    pub fn set(&mut self, kind: TypesKind) {
        self.0 = Rc::new(kind);
    }

    /// Sets the size of pointers.
    pub fn set_ptr_size(size: usize) {
        Self::PTR_SIZE.with(|ptr_size| {
            ptr_size.set(size);
        });
    }

}


impl cmp::PartialEq for Types {
    /// Compare two `Types`
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

impl fmt::Display for Types {
    /// Format `Types`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for Types {
    /// Format `Types`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl hash::Hash for Types {
    /// Hash `Types`
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
    fn type_from() {
        assert_eq!(Types::from_string("i8"), Types::i8());
        assert_eq!(Types::from_string("i16"), Types::i16());
        assert_eq!(Types::from_string("i32"), Types::i32());
        assert_eq!(Types::from_string("i64"), Types::i64());
        assert_eq!(Types::from_string("i128"), Types::i128());
        assert_eq!(Types::from_string("u1"), Types::u1());
        assert_eq!(Types::from_string("u8"), Types::u8());
        assert_eq!(Types::from_string("u16"), Types::u16());
        assert_eq!(Types::from_string("u32"), Types::u32());
        assert_eq!(Types::from_string("u64"), Types::u64());
        assert_eq!(Types::from_string("u128"), Types::u128());
        assert_eq!(Types::from_string("f16"), Types::f16());
        assert_eq!(Types::from_string("f32"), Types::f32());
        assert_eq!(Types::from_string("f64"), Types::f64());

        assert_eq!(Types::from_string("[i32; 10]"), Types::array(Types::i32(), 10));
        assert_eq!(Types::from_string("[[i32; 10]; 3]"), Types::array(Types::array(Types::i32(), 10), 3));
        assert_eq!(Types::from_string("*i32"), Types::ptr(Types::i32()));

        assert_eq!(Types::from_string("(i32, f64)"), Types::tuple(vec![Types::i32(), Types::f64()]));
        assert_eq!(Types::from_string("fn (i32, f64) -> f64"), Types::func(vec![Types::i32(), Types::f64()], Types::f64()));
        assert_eq!(Types::from_string("struct {i32, f64}"), Types::stc(vec![Types::i32(), Types::f64()]));
    }

    #[test]
    fn type_to() {
        assert_eq!(format!("{}", Types::i8()), "i8");
        assert_eq!(format!("{}", Types::i16()), "i16");
        assert_eq!(format!("{}", Types::i32()), "i32");
        assert_eq!(format!("{}", Types::i64()), "i64");
        assert_eq!(format!("{}", Types::i128()), "i128");
        assert_eq!(format!("{}", Types::u1()), "u1");
        assert_eq!(format!("{}", Types::u2()), "u2");
        assert_eq!(format!("{}", Types::u8()), "u8");
        assert_eq!(format!("{}", Types::u16()), "u16");
        assert_eq!(format!("{}", Types::u32()), "u32");
        assert_eq!(format!("{}", Types::u64()), "u64");
        assert_eq!(format!("{}", Types::u128()), "u128");
        assert_eq!(format!("{}", Types::f16()), "f16");
        assert_eq!(format!("{}", Types::f32()), "f32");
        assert_eq!(format!("{}", Types::f64()), "f64");
        assert_eq!(format!("{}", Types::f128()), "f128");
        assert_eq!(format!("{}", Types::array(Types::i32(), 10)), "[i32; 10]");
        assert_eq!(format!("{}", Types::array(Types::array(Types::i32(), 10), 3)), "[[i32; 10]; 3]");
        assert_eq!(format!("{}", Types::ptr(Types::ptr(Types::f64()))), "**f64");
        assert_eq!(format!("{}", Types::func(vec![Types::i32(), Types::f64()], Types::f64())), "fn (i32, f64) -> f64");
        assert_eq!(format!("{}", Types::tuple(vec![Types::i32(), Types::f64()])), "(i32, f64)");
        assert_eq!(format!("{}", Types::stc(vec![Types::i32(), Types::f64()])), "struct {i32, f64}");
    }

    #[test]
    fn type_eq() {
        assert_eq!(Types::i8(), Types::i8());
        assert_eq!(Types::array(Types::i32(), 6), Types::array(Types::i32(), 6));
    }

    #[test]
    fn type_size() {

        assert_eq!(Types::i8().size(), 1);
        assert_eq!(Types::i16().size(), 2);
        assert_eq!(Types::i32().size(), 4);
        assert_eq!(Types::i64().size(), 8);
        assert_eq!(Types::i128().size(), 16);
        assert_eq!(Types::u1().size(), 1);
        assert_eq!(Types::u8().size(), 1);
        assert_eq!(Types::u16().size(), 2);
        assert_eq!(Types::u32().size(), 4);
        assert_eq!(Types::u64().size(), 8);
        assert_eq!(Types::u128().size(), 16);
        assert_eq!(Types::f16().size(), 2);
        assert_eq!(Types::f32().size(), 4);
        assert_eq!(Types::f64().size(), 8);
        assert_eq!(Types::f128().size(), 16);

        assert_eq!(Types::array(Types::i32(), 10).size(), 10 * 4);
        assert_eq!(Types::array(Types::array(Types::i32(), 10), 3).size(), 3 * 10 * 4);
        assert_eq!(Types::ptr(Types::f64()).size(), mem::size_of::<usize>());

        Types::set_ptr_size(4);
        assert_eq!(Types::ptr(Types::f64()).size(), 4);
        assert_eq!(Types::array(Types::ptr(Types::i32()), 5).size(), 4 * 5);
        assert_eq!(Types::tuple(vec![Types::i32(), Types::f32()]).size(), 8);
        assert_eq!(Types::func(vec![Types::i32(), Types::f64()], Types::f64()).size(), 4);
        assert_eq!(Types::stc(vec![Types::i32(), Types::f32()]).size(), 4);
    } 


    #[test]
    fn type_scale() {
        assert_eq!(Types::i32().scale(), vec![32]);
        assert_eq!(Types::u1().scale(), vec![1]);
        assert_eq!(Types::u7().scale(), vec![7]);

        assert_eq!(Types::array(Types::i32(), 5).scale(), vec![32, 32, 32, 32, 32]);
        assert_eq!(Types::array(Types::array(Types::i32(), 5), 3).scale(), vec![160, 160, 160]);
        assert_eq!(Types::tuple(vec![Types::i32(), Types::f32()]).scale(), vec![32, 32]);
        assert_eq!(Types::stc(vec![Types::i32(), Types::f32()]).scale(), vec![64]);
        
        Types::set_ptr_size(4);
        assert_eq!(Types::ptr(Types::f64()).scale(), vec![32]);
        assert_eq!(Types::array(Types::ptr(Types::i32()), 5).scale(), vec![32, 32, 32, 32, 32]);
        assert_eq!(Types::func(vec![Types::i32(), Types::f64()], Types::f64()).scale(), vec![32]);
    }


    #[test]
    fn type_types() {
        assert_eq!(Types::i32().types(), vec![Types::i32()]);
        assert_eq!(Types::array(Types::i32(), 5).types(), vec![Types::i32(), Types::i32(), Types::i32(), Types::i32(), Types::i32()]);
        assert_eq!(Types::array(Types::array(Types::i32(), 5), 3).types(), vec![Types::array(Types::i32(), 5), Types::array(Types::i32(), 5), Types::array(Types::i32(), 5)]);
        assert_eq!(Types::tuple(vec![Types::i32(), Types::f32()]).types(), vec![Types::i32(), Types::f32()]);
        assert_eq!(Types::stc(vec![Types::i32(), Types::f32()]).types(), vec![Types::i32(), Types::f32()]);
        assert_eq!(Types::func(vec![Types::i32(), Types::f64()], Types::f64()).types(), vec![Types::i32(), Types::f64(), Types::f64()]);
    }

}