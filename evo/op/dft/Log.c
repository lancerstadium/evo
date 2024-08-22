#include "../../evo/resolver.h"
#include "../../util/math.h"
#include <math.h>

static void Log_bfloat16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = bfloat16_to_float32(px[i]);
        py[i] = float32_to_bfloat16(logf(v));
    }
}

static void Log_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(px[i]);
        py[i] = float32_to_float16(logf(v));
    }
}

static void Log_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = logf(px[i]);
}

static void Log_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = log(px[i]);
}

void op_Log_dft(node_t *nd) {
    // 1. Log init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    // 2. Log reshape
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
    // 3. Log run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            Log_float16(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Log_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Log_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Log_float64(nd);
            break;
        default:
            break;
    }
    // 4. Log exit
    return;
}