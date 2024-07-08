#include "evo.h"
#include "sys.h"
#include "op/dft/def.h"

// ==================================================================================== //
//                                     operator
// ==================================================================================== //

void op_copy(op_t *s, op_t *t) {
    if(s && t) {
        s->type = t->type;
        s->name = sys_strdup(t->name);
        s->run = t->run;
        s->is_same_shape = t->is_same_shape;
    }
}

static void op_Nop(node_t* nd) {

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
        [OP_TYPE_NOP]       = { .type = OP_TYPE_NOP         , .name = "nop"     , .run = op_Nop     },
        [OP_TYPE_ABS]       = { .type = OP_TYPE_ABS         , .name = "abs"     , .run = op_Abs_dft },
        [OP_TYPE_ADD]       = { .type = OP_TYPE_ADD         , .name = "add"     , .run = op_Add_dft },
        [OP_TYPE_RESHAPE]   = { .type = OP_TYPE_RESHAPE     , .name = "reshape" , .run = NULL       },

    }
};


// ==================================================================================== //
//                                  resolver API
// ==================================================================================== //

resolver_t* resolver_get_default() {
    return &default_resolver;
}