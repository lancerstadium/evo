#include <evo/resolver.h>
#include <evo/util/math.h>


typedef struct {
    float alpha;
} operator_pdata_t;

static void LeakyRelu_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(px[i]);
        if (v < 0)
            v *= pdat->alpha;
        py[i] = float32_to_float16(v);
    }
}

static void LeakyRelu_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

static void LeakyRelu_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

static void LeakyRelu_backward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->name);
        nd->in[0]->grad = tensor_new(name_buf, x->type);
        tensor_reshape(nd->in[0]->grad, x->ndim, x->dims);
    }
    tensor_t *delta = nd->in[0]->grad;
    uint16_t *pd = (uint16_t *)delta->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(py[i]);
        pd[i] = float32_to_float16((v > 0) ? 1 : pdat->alpha);
    }
}
static void LeakyRelu_backward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->name);
        nd->in[0]->grad = tensor_new(name_buf, x->type);
        tensor_reshape(nd->in[0]->grad, x->ndim, x->dims);
    }
    tensor_t *delta = nd->in[0]->grad;
    float *pd = (float *)delta->datas;
    float *py = (float *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        pd[i] = (py[i] > 0) ? 1 : pdat->alpha;
}
static void LeakyRelu_backward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->name);
        nd->in[0]->grad = tensor_new(name_buf, x->type);
        tensor_reshape(nd->in[0]->grad, x->ndim, x->dims);
    }
    tensor_t *delta = nd->in[0]->grad;
    double *pd = (double *)delta->datas;
    double *py = (double *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        pd[i] = (py[i] > 0) ? 1 : (double)pdat->alpha;
}


void LeakyRelu_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->alpha = node_get_attr_float(nd, "alpha", 0.01);
        nd->priv = pdat;
    }
}

void LeakyRelu_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
}

void LeakyRelu_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            LeakyRelu_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            LeakyRelu_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            LeakyRelu_forward_float64(nd);
            break;
        default:
            break;
    }
}

void LeakyRelu_backward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            LeakyRelu_backward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            LeakyRelu_backward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            LeakyRelu_backward_float64(nd);
            break;
        default:
            break;
    }
}

void LeakyRelu_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_LeakyRelu_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = LeakyRelu_init;
    nd->op->reshape     = LeakyRelu_reshape;
    nd->op->forward     = LeakyRelu_forward;
    nd->op->backward    = LeakyRelu_backward;
    nd->op->exit        = LeakyRelu_exit;
}