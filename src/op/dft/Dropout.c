#include <evo/resolver.h>

static void Dropout_bfloat16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = px[i];
}

static void Dropout_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = px[i];
}

static void Dropout_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = px[i];
}

static void Dropout_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = px[i];
}

void Dropout_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void Dropout_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout >= 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
}

void Dropout_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout >= 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BFLOAT16:
            Dropout_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Dropout_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Dropout_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Dropout_float64(nd);
            break;
        default:
            break;
    }
}

void Dropout_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Dropout_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Dropout_init;
    nd->op->reshape     = Dropout_reshape;
    nd->op->forward     = Dropout_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Dropout_exit;
}