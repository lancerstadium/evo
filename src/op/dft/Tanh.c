#include <evo/resolver.h>
#include <evo/util/math.h>
#include <math.h>

static void Tanh_foward_bfloat16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = bfloat16_to_float32(px[i]);
        py[i] = float32_to_bfloat16(tanhf(v));
    }
}

static void Tanh_foward_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(px[i]);
        py[i] = float32_to_float16(tanhf(v));
    }
}

static void Tanh_foward_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = tanhf(px[i]);
}

static void Tanh_foward_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = tanh(px[i]);
}


static void Tanh_backward_bfloat16(node_t* nd) {
    tensor_t *y = nd->in[0];
    tensor_t *d = nd->in[0]->grad;
    tensor_t *g = nd->out[0]->grad;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pd = (uint16_t *)d->datas;
    uint16_t *pg = (uint16_t *)g->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        pd[i] = float32_to_bfloat16((1.0f - bfloat16_to_float32(py[i]) * bfloat16_to_float32(py[i])) * bfloat16_to_float32(pg[i]));
}

static void Tanh_backward_float16(node_t* nd) {
    tensor_t *y = nd->out[0];
    tensor_t *d = nd->in[0]->grad;
    tensor_t *g = nd->out[0]->grad;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pd = (uint16_t *)d->datas;
    uint16_t *pg = (uint16_t *)g->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        pd[i] = float32_to_float16((1.0f - float16_to_float32(py[i]) * float16_to_float32(py[i])) * float16_to_float32(pg[i]));
}

static void Tanh_backward_float32(node_t* nd) {
    tensor_t *y = nd->out[0];
    tensor_t *d = nd->in[0]->grad;
    tensor_t *g = nd->out[0]->grad;
    float *py = (float *)y->datas;
    float *pd = (float *)d->datas;
    float *pg = (float *)g->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        pd[i] = (1.0f - py[i] * py[i]) * pg[i];
}

static void Tanh_backward_float64(node_t* nd) {
    tensor_t *y = nd->out[0];
    tensor_t *d = nd->in[0]->grad;
    tensor_t *g = nd->out[0]->grad;
    double *py = (double *)y->datas;
    double *pd = (double *)d->datas;
    double *pg = (double *)g->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        pd[i] = (1.0 - py[i] * py[i]) * pg[i];
        
}

void Tanh_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void Tanh_reshape(node_t *nd) {
    if (!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
}

void Tanh_forward(node_t *nd) {
    if (!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BFLOAT16:
            Tanh_foward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Tanh_foward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Tanh_foward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Tanh_foward_float64(nd);
            break;
        default:
            break;
    }
}

void Tanh_backward(node_t *nd) {
    if (!nd || !nd->in || !nd->out) return;
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->in[0]->name);
        nd->in[0]->grad = tensor_new(name_buf, nd->in[0]->type);
        tensor_reshape(nd->in[0]->grad, nd->in[0]->ndim, nd->in[0]->dims);
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BFLOAT16:
            Tanh_backward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Tanh_backward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Tanh_backward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Tanh_backward_float64(nd);
            break;
        default:
            break;
    }
}

void Tanh_exit(node_t *nd) {
    if (!nd || !nd->in || !nd->out) return;
    return;
}

void op_Tanh_dft(node_t *nd) {
    if (!nd || !nd->op) return;
    nd->op->init        = Tanh_init;
    nd->op->reshape     = Tanh_reshape;
    nd->op->forward     = Tanh_forward;
    nd->op->backward    = Tanh_backward;
    nd->op->exit        = Tanh_exit;
}