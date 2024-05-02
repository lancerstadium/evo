



// ============================================================================== //
//                                 Use Mods
// ============================================================================== //





pub trait DTypeKindTrait {
    type W;
}


impl<T> DTypeKindTrait for T {
    type W = usize;
}

/// DType: Data Type
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct DType<T: DTypeKindTrait> {
    pub width: T::W,
}


impl<T: DTypeKindTrait> DType<T> {
    pub fn new(width: T::W) -> Self {
        Self { width }
    }
}



#[cfg(test)]
mod dtype_test {
    use super::*;

    #[test]
    fn dtype_gen() {
        let dt = DType::<u32>::new(32);
        print!("{:?}", dt);
    }
}