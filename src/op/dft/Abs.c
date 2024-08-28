#include <evo/resolver.h>
#include <evo/util/math.h>
#include <math.h>

static void Abs_int8(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_int16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int16_t *px = (int16_t *)x->datas;
    int16_t *py = (int16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_int32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_int64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int64_t *px = (int64_t *)x->datas;
    int64_t *py = (int64_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint8(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    uint8_t *py = (uint8_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint32_t *px = (uint32_t *)x->datas;
    uint32_t *py = (uint32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint64_t *px = (uint64_t *)x->datas;
    uint64_t *py = (uint64_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_bfloat16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = bfloat16_to_float32(px[i]);
        py[i] = float32_to_bfloat16(fabsf(v));
    }
}

static void Abs_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(px[i]);
        py[i] = float32_to_float16(fabsf(v));
    }
}

static void Abs_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = fabsf(px[i]);
}

static void Abs_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = fabs(px[i]);
}

void op_Abs_dft(node_t *nd) {
    // 1. Abs init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    // 2. Abs reshape
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
    // 3. Abs run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT8:      Abs_int8(nd); break;
        case TENSOR_TYPE_INT16:     Abs_int16(nd); break;
        case TENSOR_TYPE_INT32:     Abs_int32(nd); break;
        case TENSOR_TYPE_INT64:     Abs_int64(nd); break;
        case TENSOR_TYPE_UINT8:     Abs_uint8(nd); break;
        case TENSOR_TYPE_UINT16:    Abs_uint16(nd); break;
        case TENSOR_TYPE_UINT32:    Abs_uint32(nd); break;
        case TENSOR_TYPE_UINT64:    Abs_uint64(nd); break;
        case TENSOR_TYPE_FLOAT16:   Abs_float16(nd); break;
        case TENSOR_TYPE_BFLOAT16:  Abs_bfloat16(nd); break;
        case TENSOR_TYPE_FLOAT32:   Abs_float32(nd); break;
        case TENSOR_TYPE_FLOAT64:   Abs_float64(nd); break;
        default: break;
    }
    // 4. Abs exit
    return;
}