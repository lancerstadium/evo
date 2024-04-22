//! bits operation macros


/// set bit: set to one
#[macro_export]
macro_rules! bit_set {
    ($value:expr, $n:expr) => {
        $value |= (1 << $n)
    };
}


/// clear bit: set to zero
#[macro_export]
macro_rules! bit_clr {
    ($value:expr, $n:expr) => {
        $value &= !(1 << $n)
    };
}


/// bit equal 1
#[macro_export]
macro_rules! bit_eq1 {
    ($value:expr, $n:expr) => {
        ($value & (1 << $n)) != 0
    };
}


/// bit equal 0
#[macro_export]
macro_rules! bit_eq0 {
    ($value:expr, $n:expr) => {
        ($value & (1 << $n)) == 0
    };
}


/// bit shift left
#[macro_export]
macro_rules! bit_shl {
    ($value:expr, $n:expr) => {
        $value << $n
    };
}


/// bit shift right
#[macro_export]
macro_rules! bit_shr {
    ($value:expr, $n:expr) => {
        $value >> $n
    };
}


/// bit rotate left
#[macro_export]
macro_rules! bit_rotl {
    ($value:expr, $n:expr) => {
        $value << $n | $value >> (64 - $n)
    };
}


/// bit rotate right
#[macro_export]
macro_rules! bit_rotr {
    ($value:expr, $n:expr) => {
        $value >> $n | $value << (64 - $n)
    };
}


/// bit not
#[macro_export]
macro_rules! bit_not {
    ($value:expr) => {
        !$value
    };
}


/// bit and
#[macro_export]
macro_rules! bit_and {
    ($lhs:expr, $rhs:expr) => {
        $lhs & $rhs
    };
}


/// bit or
#[macro_export]
macro_rules! bit_or {
    ($lhs:expr, $rhs:expr) => {
        $lhs | $rhs
    };
}


/// bit xor
#[macro_export]
macro_rules! bit_xor {
    ($lhs:expr, $rhs:expr) => {
        $lhs ^ $rhs
    };
}

