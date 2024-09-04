#include <evo/resolver.h>
#include <evo/util/math.h>


void Shape_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
}

void Shape_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    y->type = TENSOR_TYPE_INT64;
    tensor_reshape(y, 1, (int[]){x->ndim});
}

void Shape_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(nd->in[0]->type == TENSOR_TYPE_UNDEFINED) return;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    int64_t* py = (int64_t*)y->datas;
    size_t i, l;

    for (i = 0, l = MIN(y->ndata, (size_t)x->ndim); i < l; i++)
        py[i] = x->dims[i];
}

void Shape_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Shape_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Shape_init;
    nd->op->reshape     = Shape_reshape;
    nd->op->forward     = Shape_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Shape_exit;
}