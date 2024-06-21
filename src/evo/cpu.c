

// ==================================================================================== //
//                                    evo: CPUState
// ==================================================================================== //

#define CPUState_def(W, A)      \
    typedef struct {            \
        W stack[EVO_STACK_CAP]; \
        u64 stack_size;         \
        bool halt;              \
    }

