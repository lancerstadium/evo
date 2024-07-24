#include "../../core/resolver.h"
#include "../../util/math.h"


typedef struct {
    float alpha;
} operator_pdata_t;

static void LeakyRelu_float16(node_t *nd) {
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

static void LeakyRelu_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

static void LeakyRelu_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

void op_LeakyRelu_dft(node_t *nd) {
    // 1. LeakyRelu init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->alpha = node_get_attr_float(nd, "alpha", 0.01);
        nd->priv = pdat;
    }
    // 2. LeakyRelu reshape
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
    // 3. LeakyRelu run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            LeakyRelu_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            LeakyRelu_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            LeakyRelu_float64(nd);
            break;
        default:
            break;
    }
    // 4. LeakyRelu exit
    if (pdat)
        free(pdat);
    return;
}