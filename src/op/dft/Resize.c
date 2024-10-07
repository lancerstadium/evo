#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>
#include <evo/dev/cpu/def.h>

// ref: https://lankning.github.io/2023/02/13/%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/ONNX%E7%AE%97%E5%AD%90%E5%88%86%E6%9E%90-Resize
// ref: https://onnx.ai/onnx/operators/onnx__Resize.html

typedef struct {
    char* coordinate_transformation_mode;       /* half_pixel, half_pixel_symmetric, pytorch_half_pixel, align_corners, asymmetric, tf_crop_and_resize */
    float cubic_coeff_a;                        /* -0.75 */
    int64_t exclude_outside;                    /* 0 */
    float extrapolation_value;                  /* 0.0 */
    char* keep_aspect_ration_policy;            /* stretch, not_larger, not_smaller */
    char* mode;                                 /* nearest, linear, cubic */
    char* nearest_mode;                         /* round_prefer_floor, round_prefer_ceil, floor, ceil */
} operator_pdata_t;

void Resize_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->mode = node_get_attr_string(nd, "mode", "nearest");
        nd->priv = pdat;
    }
}

void Resize_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 4) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t *x = nd->in[0];
    tensor_t *sc = nd->in[2];
    tensor_t *y = nd->out[0];
    y->type = x->type;
    int new_dims[x->ndim];
    float* p = (float*)sc->datas;
    for (int i = 0; i < x->ndim; i++) {
        new_dims[i] = (float)x->dims[i] * (p[i]);
    }
    tensor_reshape(y, x->ndim, new_dims);
}

void Resize_forward_uint8(node_t* nd) {
    operator_pdata_t* pdat = nd->priv;
    tensor_t *x = nd->in[0];
    float *sc = nd->in[2]->datas;
    tensor_t *y = nd->out[0];
    switch(shash(pdat->mode)) {
        case 0x09fa48d7: Resize_nearest_uint8_cpu(x->datas, y->datas, x->dims[0], x->dims[1], x->dims[2], x->dims[3], 1, sc[3], true); break; /* nearest */
        case 0x8320f06b: break; /* linear */
        /* cubic */
        default: break;
    }
}

void Resize_forward_float32(node_t* nd) {
    operator_pdata_t* pdat = nd->priv;
    tensor_t *x = nd->in[0];
    float *sc = nd->in[2]->datas;
    tensor_t *y = nd->out[0];
    switch(shash(pdat->mode)) {
        case 0x09fa48d7: Resize_nearest_float32_cpu(x->datas, y->datas, x->dims[0], x->dims[1], x->dims[2], x->dims[3], 1, sc[3], true); break; /* nearest */
        case 0x8320f06b: break; /* linear */
        /* cubic */
        default: break;
    }
}

void Resize_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 4) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }

    switch (nd->in[0]->type) {
        case TENSOR_TYPE_UINT8:
            // stbir_resize_uint8_linear((const unsigned char*)x->datas, x->dims[3], x->dims[2], 0, (unsigned char*)y->datas, y->dims[3], y->dims[2], 0, y->dims[1]);
            Resize_forward_uint8(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            // stbir_resize_float_linear((const float*)x->datas, x->dims[3], x->dims[2], 0, (float*)y->datas, y->dims[3], y->dims[2], 0, y->dims[1]);
            Resize_forward_float32(nd);
            break;
        default:
            break;
    }
}

void Resize_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat) {
        free(pdat);
    }
    nd->priv = NULL;
    return;
}

void op_Resize_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Resize_init;
    nd->op->reshape     = Resize_reshape;
    nd->op->forward     = Resize_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Resize_exit;
}