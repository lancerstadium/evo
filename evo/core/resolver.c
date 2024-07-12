#include "resolver.h"
#include "../util/log.h"

// ==================================================================================== //
//                                     operator
// ==================================================================================== //

static void op_Nop(node_t* nd) {
    LOG_WARN("Nop\n");
}

// ==================================================================================== //
//                                  resolver: default
// ==================================================================================== //

static void* resolver_init_dft() {
    return NULL;
}

static void resolver_release_dft(void* rctx) {
}

static resolver_t default_resolver = {
    .name = "default",
    .init = resolver_init_dft,
    .release = resolver_release_dft,

    .op_tbl = (op_t[]){
        [OP_TYPE_NOP]       = { .type = OP_TYPE_NOP         , .name = "Nop"     , .run = op_Nop             },
        [OP_TYPE_ABS]       = { .type = OP_TYPE_ABS         , .name = "Abs"     , .run = op_Abs_dft         },
        [OP_TYPE_ADD]       = { .type = OP_TYPE_ADD         , .name = "Add"     , .run = op_Add_dft         },
        [OP_TYPE_CONV]      = { .type = OP_TYPE_CONV        , .name = "Conv"    , .run = op_Conv_dft        },
        [OP_TYPE_MAT_MUL]   = { .type = OP_TYPE_MAT_MUL     , .name = "MatMul"  , .run = op_MatMul_dft      },
        [OP_TYPE_MAX_POOL]  = { .type = OP_TYPE_MAX_POOL    , .name = "MaxPool" , .run = op_MaxPool_dft     },
        [OP_TYPE_RELU]      = { .type = OP_TYPE_RELU        , .name = "Relu"    , .run = op_Relu_dft        },
        [OP_TYPE_RESHAPE]   = { .type = OP_TYPE_RESHAPE     , .name = "Reshape" , .run = op_Reshape_dft     },
    }
};


// ==================================================================================== //
//                                  resolver API
// ==================================================================================== //

resolver_t* resolver_get_default() {
    return &default_resolver;
}