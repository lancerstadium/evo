#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>
#include "../../vis/stb_image_resize2.h"

typedef struct {
    char* mode;
} operator_pdata_t;

void Resize_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type != TENSOR_TYPE_FLOAT32) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        switch(shash(node_get_attr_string(nd, "mode", "nearest"))) {
            case 0x09fa48d7: break; /* nearest */
            case 0x8320f06b: break; /* bilinear */
            default: break;
        }
        nd->priv = pdat;
    }
}

void Resize_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *sc = nd->in[1];
    tensor_t *y = nd->out[0];
    y->type = x->type;
    int new_dims[x->ndim];
    float* p = (float*)sc->datas;
    for (int i = 0; i < x->ndim; i++) {
        new_dims[i] = x->dims[i] * (p[i]);
    }
    tensor_reshape(y, x->ndim, new_dims);
}

void Resize_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_UINT8:
            stbir_resize_uint8_linear((const unsigned char*)x->datas, x->dims[3], x->dims[2], 0, (unsigned char*)y->datas, y->dims[3], y->dims[2], 0, y->dims[1]);
            break;
        case TENSOR_TYPE_FLOAT32:
            stbir_resize_float_linear((const float*)x->datas, x->dims[3], x->dims[2], 0, (float*)y->datas, y->dims[3], y->dims[2], 0, y->dims[1]);
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