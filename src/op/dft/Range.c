#include <evo/resolver.h>
#include <evo/util/math.h>
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

static void Range_forward_int16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int16_t *py = (int16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_forward_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int32_t *py = (int32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_forward_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int64_t *py = (int64_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

static void Range_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    double *py = (double *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pdat->start + (pdat->delta * i);
}

void Range_init(node_t *nd){
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    pdat->start = tensor_get_value(nd->in[0]->datas, nd->in[0]->type);
    pdat->limit = tensor_get_value(nd->in[1]->datas, nd->in[1]->type);
    pdat->delta = tensor_get_value(nd->in[2]->datas, nd->in[2]->type);
    nd->priv = pdat;
}

void Range_reshape(node_t *nd){
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    int ndim;
    ndim = fmax(ceil((pdat->limit - pdat->start) / pdat->delta), 0);
    y->type = nd->in[0]->type;
    tensor_reshape(y, 1, (int[]){ndim});
}

void Range_forward(node_t *nd){
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT16:
            Range_forward_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Range_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Range_forward_int64(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Range_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Range_forward_float64(nd);
            break;
        default:
            break;
    }
}

void Range_exit(node_t *nd){
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Range_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Range_init;
    nd->op->reshape     = Range_reshape;
    nd->op->forward     = Range_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Range_exit;
}