#include <evo/resolver.h>
#include <evo/util/math.h>
#include <float.h>
#include <string.h>

union onnx_scalar_t {
    uint8_t v_bool;
    int8_t v_int8;
    int16_t v_int16;
    int32_t v_int32;
    int64_t v_int64;
    uint8_t v_uint8;
    uint16_t v_uint16;
    uint32_t v_uint32;
    uint64_t v_uint64;
    uint16_t v_bfloat16;
    uint16_t v_float16;
    float v_float32;
    double v_float64;
    struct {
        float real;
        float imaginary;
    } v_complex64;
    struct {
        double real;
        double imaginary;
    } v_complex128;
};

typedef struct {
    union onnx_scalar_t *pmin;
    union onnx_scalar_t *pmax;
} operator_pdata_t;

static void Clip_forward_int16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int16_t *px = (int16_t *)x->datas;
    int16_t *py = (int16_t *)y->datas;
    int16_t minv = pdat->pmin ? pdat->pmin->v_int16 : INT16_MIN;
    int16_t maxv = pdat->pmax ? pdat->pmax->v_int16 : INT16_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int32_t minv = pdat->pmin ? pdat->pmin->v_int32 : INT32_MIN;
    int32_t maxv = pdat->pmax ? pdat->pmax->v_int32 : INT32_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int64_t *px = (int64_t *)x->datas;
    int64_t *py = (int64_t *)y->datas;
    int64_t minv = pdat->pmin ? pdat->pmin->v_int64 : INT64_MIN;
    int64_t maxv = pdat->pmax ? pdat->pmax->v_int64 : INT64_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_uint8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    uint8_t *py = (uint8_t *)y->datas;
    uint8_t minv = pdat->pmin ? pdat->pmin->v_uint8 : 0;
    uint8_t maxv = pdat->pmax ? pdat->pmax->v_uint8 : UINT8_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_uint16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t minv = pdat->pmin ? pdat->pmin->v_uint16 : 0;
    uint16_t maxv = pdat->pmax ? pdat->pmax->v_uint16 : UINT16_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_uint32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint32_t *px = (uint32_t *)x->datas;
    uint32_t *py = (uint32_t *)y->datas;
    uint32_t minv = pdat->pmin ? pdat->pmin->v_uint32 : 0;
    uint32_t maxv = pdat->pmax ? pdat->pmax->v_uint32 : UINT32_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_uint64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint64_t *px = (uint64_t *)x->datas;
    uint64_t *py = (uint64_t *)y->datas;
    uint64_t minv = pdat->pmin ? pdat->pmin->v_uint64 : 0;
    uint64_t maxv = pdat->pmax ? pdat->pmax->v_uint64 : UINT64_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_bfloat16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float minv = bfloat16_to_float32(pdat->pmin ? pdat->pmin->v_bfloat16 : float32_to_bfloat16(-FLT_MAX));
    float maxv = bfloat16_to_float32(pdat->pmax ? pdat->pmax->v_bfloat16 : float32_to_bfloat16(+FLT_MAX));
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = bfloat16_to_float32(px[i]);
        if (v < minv)
            v = minv;
        else if (px[i] > maxv)
            v = maxv;
        else
            v = px[i];
        py[i] = float32_to_bfloat16(v);
    }
}

static void Clip_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float minv = float16_to_float32(pdat->pmin ? pdat->pmin->v_float16 : float32_to_float16(-FLT_MAX));
    float maxv = float16_to_float32(pdat->pmax ? pdat->pmax->v_float16 : float32_to_float16(+FLT_MAX));
    float v;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        v = float16_to_float32(px[i]);
        if (v < minv)
            v = minv;
        else if (px[i] > maxv)
            v = maxv;
        else
            v = px[i];
        py[i] = float32_to_float16(v);
    }
}

static void Clip_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    float minv = pdat->pmin ? pdat->pmin->v_float32 : -FLT_MAX;
    float maxv = pdat->pmax ? pdat->pmax->v_float32 : +FLT_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

static void Clip_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    double minv = pdat->pmin ? pdat->pmin->v_float64 : -DBL_MAX;
    double maxv = pdat->pmax ? pdat->pmax->v_float64 : +DBL_MAX;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if (px[i] < minv)
            py[i] = minv;
        else if (px[i] > maxv)
            py[i] = maxv;
        else
            py[i] = px[i];
    }
}

void Clip_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if(pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->pmin = NULL;
        pdat->pmax = NULL;
        nd->priv = pdat;
    }
}

void Clip_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *x = nd->in[0];
    for (int i = 1; i < nd->nin; i++) {
        if (nd->in[i]->ndim == 0) {
            if (strcmp(nd->in[i]->name, "min") == 0)
                pdat->pmin = (union onnx_scalar_t *)nd->in[i]->datas;
            else if (strcmp(nd->in[i]->name, "max") == 0)
                pdat->pmax = (union onnx_scalar_t *)nd->in[i]->datas;
        }
    }
    tensor_reshape_ident(y, x, x->type);
}

void Clip_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT16:     Clip_forward_int16(nd); break;
        case TENSOR_TYPE_INT32:     Clip_forward_int32(nd); break;
        case TENSOR_TYPE_INT64:     Clip_forward_int64(nd); break;
        case TENSOR_TYPE_UINT8:     Clip_forward_uint8(nd); break;
        case TENSOR_TYPE_UINT16:    Clip_forward_uint16(nd); break;
        case TENSOR_TYPE_UINT32:    Clip_forward_uint32(nd); break;
        case TENSOR_TYPE_UINT64:    Clip_forward_uint64(nd); break;
        case TENSOR_TYPE_FLOAT16:   Clip_forward_float16(nd); break;
        case TENSOR_TYPE_BFLOAT16:  Clip_forward_bfloat16(nd); break;
        case TENSOR_TYPE_FLOAT32:   Clip_forward_float32(nd); break;
        case TENSOR_TYPE_FLOAT64:   Clip_forward_float64(nd); break;
        default: break;
    }
}

void Clip_backward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->in[0]->name);
        nd->in[0]->grad = tensor_new(name_buf, nd->in[0]->type);
        tensor_reshape(nd->in[0]->grad, nd->in[0]->ndim, nd->in[0]->dims);
    }
    switch (nd->in[0]->type) {
        // case TENSOR_TYPE_INT32:     Clip_backward_int32(nd); break;
        // case TENSOR_TYPE_INT64:     Clip_backward_int64(nd); break;
        // case TENSOR_TYPE_UINT32:    Clip_backward_uint32(nd); break;
        // case TENSOR_TYPE_UINT64:    Clip_backward_uint64(nd); break;
        // case TENSOR_TYPE_FLOAT32:   Clip_backward_float32(nd); break;
        // case TENSOR_TYPE_FLOAT64:   Clip_backward_float64(nd); break;
        default: break;
    }
}

void Clip_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Clip_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Clip_init;
    nd->op->reshape     = Clip_reshape;
    nd->op->forward     = Clip_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Clip_exit;
}