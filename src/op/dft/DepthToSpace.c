#include <evo/resolver.h>



void DepthToSpace_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void DepthToSpace_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->ndim != 4
        || nd->in[0]->layout == 1) {
        return;
    }
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    y->type = x->type;

}

void DepthToSpace_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->ndim != 4
        || nd->in[0]->layout == 1) {
        return;
    }

}

void DepthToSpace_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_DepthToSpace_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = DepthToSpace_init;
    nd->op->reshape     = DepthToSpace_reshape;
    nd->op->forward     = DepthToSpace_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = DepthToSpace_exit;
}