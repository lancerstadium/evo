#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>
#include "../../vis/stb_image_resize2.h"

typedef struct {
    char* mode;
} operator_pdata_t;


void op_Resize_dft(node_t* nd) {
    // 1. Upsample init
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
    // 2. Upsample reshape
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
    
    // 3. Upsample run
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
    // 4. Upsample exit
    if (pdat) {
        free(pdat);
    }
    nd->priv = NULL;
    return;
}