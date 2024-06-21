#include <evo/typ.h>
#include <sob/sob.h>

#define Width_OP(W, T, OP)          CONCAT3(W##_, T##_, OP)
#define Width_OP_def(W, T, OP)      UNUSED Width_OP(W, T, OP)

#define Width_def(W, T)                \
    W Width_OP_def(W, T, new)(T val) { \
        return (W){                    \
            .as_##T = val};            \
    }

Width_def(Word, u32);
Width_def(Word, i32);
Width_def(Word, f32);
#if EVO_WORD_SIZE == 32
Width_def(Word, ptr);
#endif

Width_def(Dword, u64);
Width_def(Dword, i64);
Width_def(Dword, f64);
#if EVO_WORD_SIZE == 64
Width_def(Dword, ptr);
#endif

