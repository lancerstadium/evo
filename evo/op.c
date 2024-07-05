#include "evo.h"
#include "dft/op.h"


// ==================================================================================== //
//                                  resolver: default
// ==================================================================================== //


resolver_t default_resolver = {
    .name = "default",
    .init = resolver_init_dft,
    .release = resolver_release_dft,

    .op_tbl = (op_t[]){
        [OP_TYPE_ABS] = { .type = OP_TYPE_ABS, .name = "abs", .run = op_Abs_dft },
        [OP_TYPE_ADD] = { .type = OP_TYPE_ADD, .name = "add", .run = op_Add_dft },
    }
};