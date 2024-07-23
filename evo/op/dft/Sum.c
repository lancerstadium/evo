#include "../../core/resolver.h"
#include "../../util/math.h"

static void Sum_bfloat16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *px;
    float sum;
    int j;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        for (j = 0, sum = 0; j < nd->nin; j++) {
            x = nd->in[j];
            px = tensor_broadcast_map_address(x, y, i);
            sum += bfloat16_to_float32(*px);
        }
        py[i] = float32_to_bfloat16(sum);
    }
}

static void Sum_float16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *px;
    float sum;
    int j;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        for (j = 0, sum = 0; j < nd->nin; j++) {
            x = nd->in[j];
            px = tensor_broadcast_map_address(x, y, i);
            sum += float16_to_float32(*px);
        }
        py[i] = float32_to_float16(sum);
    }
}

static void Sum_float32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x;
    float *py = (float *)y->datas;
    float *px;
    float sum;
    int j;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        for (j = 0, sum = 0; j < nd->nin; j++) {
            x = nd->in[j];
            px = tensor_broadcast_map_address(x, y, i);
            sum += *px;
        }
        py[i] = sum;
    }
}

static void Sum_float64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x;
    double *py = (double *)y->datas;
    double *px;
    double sum;
    int j;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        for (j = 0, sum = 0; j < nd->nin; j++) {
            x = nd->in[j];
            px = tensor_broadcast_map_address(x, y, i);
            sum += *px;
        }
        py[i] = sum;
    }
}

void op_Sum_dft(node_t *nd) {
    // 1. Sum init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin >= 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    // 2. Sum reshape
    tensor_t *y = nd->out[0];
    int i;
    if (!tensor_reshape_ident(y, nd->in[0], nd->in[0]->type))
        return;
    for (i = 1; i < nd->nin; i++) {
        if (!tensor_reshape_multi_broadcast(y, y, nd->in[i], y->type))
            return;
    }
    // 3. Sum run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BFLOAT16:
            Sum_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Sum_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Sum_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Sum_float64(nd);
            break;
        default:
            break;
    }
    // 4. Sum exit
    return;
}