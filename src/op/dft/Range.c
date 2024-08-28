#include "../../evo/resolver.h"
#include <util/math.h>
#include <math.h>

typedef struct {
    double start;
    double limit;
    double delta;
} operator_pdata_t;

static double tensor_get_value(void *p, tensor_type_t type) {
    double v;

    switch (type) {
        case TENSOR_TYPE_BOOL:
            v = *((uint8_t *)p);
            break;
        case TENSOR_TYPE_INT8:
            v = *((int8_t *)p);
            break;
        case TENSOR_TYPE_INT16:
            v = *((int16_t *)p);
            break;
        case TENSOR_TYPE_INT32:
            v = *((int32_t *)p);
            break;
        case TENSOR_TYPE_INT64:
            v = *((int64_t *)p);
            break;
        case TENSOR_TYPE_UINT8:
            v = *((uint8_t *)p);
            break;
        case TENSOR_TYPE_UINT16:
            v = *((uint16_t *)p);
            break;
        case TENSOR_TYPE_UINT32:
            v = *((uint32_t *)p);
            break;
        case TENSOR_TYPE_UINT64:
            v = *((uint64_t *)p);
            break;
        case TENSOR_TYPE_BFLOAT16:
            v = bfloat16_to_float32(*((uint16_t *)p));
            break;
        case TENSOR_TYPE_FLOAT16:
            v = float16_to_float32(*((uint16_t *)p));
            break;
        case TENSOR_TYPE_FLOAT32:
            v = *((float *)p);
            break;
        case TENSOR_TYPE_FLOAT64:
            v = *((double *)p);
            break;
        default:
            v = 0;
            break;
    }
    return v;
}

static void Range_int16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int16_t *py = (int16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int32_t *py = (int32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int64_t *py = (int64_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    double *py = (double *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

void op_Range_dft(node_t *nd) {
    // 1. Range init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 3) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    // 2. Range reshape
    tensor_t *y = nd->out[0];
    int ndim;
    pdat->start = tensor_get_value(nd->in[0]->datas, nd->in[0]->type);
    pdat->limit = tensor_get_value(nd->in[1]->datas, nd->in[1]->type);
    pdat->delta = tensor_get_value(nd->in[2]->datas, nd->in[2]->type);
    ndim = fmax(ceil((pdat->limit - pdat->start) / pdat->delta), 0);
    y->type = nd->in[0]->type;
    tensor_reshape(y, 1, (int[]){ndim});
    // 3. Range run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT16:
            Range_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Range_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Range_int64(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Range_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Range_float64(nd);
            break;
        default:
            break;
    }
    // 4. Range exit
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}