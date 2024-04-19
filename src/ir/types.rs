//! `evo::ir::types`: Types' definition in the IR
//! 
//! ## Description
//! Every `evo-ir` value and function should have a type.
//! A type can be any of the following:
//! 1. `IntType`: integer type such as `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`.
//! 2. `FloatType`: floating point type such as `f16`, `f32`, `f64`, `f128`.
//! 3. `PtrType`: pointer type stored as a value's address.
//! 4. `ArrayType`: array type, a list of values of the same type.
//! 5. `FuncType`: function type, a list of argument types and return type.
//! 



// ============================================================================== //
//                                 Use Mods
// ============================================================================== //

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::{cmp, fmt, hash, mem};




// ============================================================================== //
//                               types::TypeKind
// ============================================================================== //

/// `TypeKind`: evo-ir enum types
#[derive(Hash, Clone, PartialEq, Eq)]
enum TypeKind {
    // Integer type
    I8, I16, I32, I64, I128, U8, U16, U32, U64, U128,

    // Float type
    F16, F32, F64, F128,

    // Pointer type
    Ptr(Type),

    // Array type
    Array(Vec<Type>, usize),

    // Function type
    Func(Vec<Type>, Type),

}



// ============================================================================== //
//                                 types::Type
// ============================================================================== //

/// `Type`: evo-ir type
#[derive(Clone, Eq)]
pub struct Type(Rc<TypeKind>);


impl Type {

    // Init type pool.
    thread_local! {
        /// Pool of all created types.
        static POOL: RefCell<HashMap<TypeKind, Type>> = RefCell::new(HashMap::new());
        /// Size of pointers.
        static PTR_SIZE: Cell<usize> = Cell::new(mem::size_of::<*const ()>());
    }

    // ==================== Type.get ===================== //

    /// Returns a type by the given `TypeKind`.
    pub fn get(type_data: TypeKind) -> Type {
        Self::POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.get(&type_data).cloned().unwrap_or_else(|| {
                let v = Self(Rc::new(type_data.clone()));
                pool.insert(type_data, v.clone());
                v
            })
        })
    }
    
    /// Returns an `i8` type.
    pub fn get_i8() -> Type {
        Type::get(TypeKind::I8)
    }

    /// Returns an `i16` type.
    pub fn get_i16() -> Type {
        Type::get(TypeKind::I16)
    }

    /// Returns an `i32` type.
    pub fn get_i32() -> Type {
        Type::get(TypeKind::Int32)
    }

    /// Returns an `i64` type.
    pub fn get_i64() -> Type {
        Type::get(TypeKind::I64)
    }

    /// Returns an `i128` type.
    pub fn get_i128() -> Type {
        Type::get(TypeKind::I128)
    }

    /// Returns an `u8` type.
    pub fn get_u8() -> Type {
        Type::get(TypeKind::U8)
    }

    /// Returns an `u16` type.
    pub fn get_u16() -> Type {
        Type::get(TypeKind::U16)
    }

    /// Returns an `u32` type.
    pub fn get_u32() -> Type {
        Type::get(TypeKind::U32)
    }

    /// Returns an `u64` type.
    pub fn get_u64() -> Type {
        Type::get(TypeKind::U64)
    }

    /// Returns an `u128` type.
    pub fn get_u128() -> Type {
        Type::get(TypeKind::U128)
    }

    /// Returns an `f16` type.
    pub fn get_f16() -> Type {
        Type::get(TypeKind::F16)
    }

    /// Returns an `f32` type.
    pub fn get_f32() -> Type {
        Type::get(TypeKind::F32)
    }

    /// Returns an `f64` type.
    pub fn get_f64() -> Type {
        Type::get(TypeKind::F64)
    }

    /// Returns an `f128` type.
    pub fn get_f128() -> Type {
        Type::get(TypeKind::F128)
    }

    /// Returns an `ptr` type.
    pub fn get_ptr() -> Type {
        Type::get(TypeKind::Ptr(Type::get_ptr_type()))
    }

    /// Returns an `array` type.
    pub fn get_array(ty: Type, size: usize) -> Type {
        Type::get(TypeKind::Array(vec![ty], size))
    }

    /// Returns an `func` type.
    pub fn get_func(args: Vec<Type>, ret: Type) -> Type {
        Type::get(TypeKind::Func(args, ret))
    }

    // ==================== Type.set ===================== //

}
