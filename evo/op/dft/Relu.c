#include "../../log.h"
#include "../../math.h"
#include "../../resolver.h"

static void Relu_int8(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_int16(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    int16_t *px = (int16_t *)x->datas;
    int16_t *py = (int16_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_int32(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_int64(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    int64_t *px = (int64_t *)x->datas;
    int64_t *py = (int64_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}
static void Relu_bfloat16(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
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
static void Relu_float16(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
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

static void Relu_float32(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_float64(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

void op_Relu_dft(node_t *nd) {
    // Relu init
    if (!nd || !nd->input_tensors || nd->input_tensors[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->ninput = 1) || !(nd->noutput == 1) || (nd->input_tensors[0]->ndim == 0)) {
        return;
    }
    // Relu reshape
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    tensor_reshape_ident(y, x, x->type);
    // Relu run
    switch (nd->input_tensors[0]->type) {
        case TENSOR_TYPE_INT8: Relu_int8(nd); break;
        case TENSOR_TYPE_INT16: Relu_int16(nd); break;
        case TENSOR_TYPE_INT32: Relu_int32(nd); break;
        case TENSOR_TYPE_INT64: Relu_int64(nd); break;
        case TENSOR_TYPE_BFLOAT16: Relu_bfloat16(nd); break;
        case TENSOR_TYPE_FLOAT16: Relu_float16(nd); break;
        case TENSOR_TYPE_FLOAT32: Relu_float32(nd); break;
        case TENSOR_TYPE_FLOAT64: Relu_float64(nd); break;
        default: break;
    }

    // Relu exit
    return;
}