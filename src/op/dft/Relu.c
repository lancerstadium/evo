#include <evo/util/log.h>
#include <evo/util/math.h>
#include <evo/resolver.h>

static void Relu_forward_int8(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_backward_int8(node_t *nd) {
    (void)nd;
}
static void Relu_forward_int16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int16_t *px = (int16_t *)x->datas;
    int16_t *py = (int16_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_forward_int32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_forward_int64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int64_t *px = (int64_t *)x->datas;
    int64_t *py = (int64_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_forward_bfloat16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = bfloat16_to_float32(px[i]);
        if (v < 0)
            v = 0;
        py[i] = float32_to_bfloat16(v);
    }
}
static void Relu_forward_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(px[i]);
        if (v < 0)
            v = 0;
        py[i] = float32_to_float16(v);
    }
}

static void Relu_forward_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_forward_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

void Relu_init(node_t *nd) {
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    } 
}

void Relu_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
}

void Relu_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT8:      Relu_forward_int8(nd); break;
        case TENSOR_TYPE_INT16:     Relu_forward_int16(nd); break;
        case TENSOR_TYPE_INT32:     Relu_forward_int32(nd); break;
        case TENSOR_TYPE_INT64:     Relu_forward_int64(nd); break;
        case TENSOR_TYPE_BFLOAT16:  Relu_forward_bfloat16(nd); break;
        case TENSOR_TYPE_FLOAT16:   Relu_forward_float16(nd); break;
        case TENSOR_TYPE_FLOAT32:   Relu_forward_float32(nd); break;
        case TENSOR_TYPE_FLOAT64:   Relu_forward_float64(nd); break;
        default: break;
    }
}

void Relu_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Relu_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Relu_init;
    nd->op->reshape     = Relu_reshape;
    nd->op->forward     = Relu_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Relu_exit;
}