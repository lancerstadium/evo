#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>

typedef struct {
    int64_t axis;               // 1
    int64_t block_size;         // 0
} operator_pdata_t;


static void DequantizeLinear_forward_int8(node_t* nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_t* c = nd->nin >= 3 ? nd->in[2] : NULL;
    switch(y->type) {
        case TENSOR_TYPE_FLOAT32: {
            float* sc = b->datas;
            if(b->ndata == 1) {
                Dequant_asymmetric_int8_to_float32_cpu(a->datas, y->datas, a->ndata, sc[0], (c ? ((int8_t*)(c->datas))[0] : 0));
            }
            break;
        }
        default: break;
    }
}


void DequantizeLinear_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->axis = node_get_attr_int(nd, "axis", 1);
        pdat->block_size = node_get_attr_int(nd, "block_size", 0);
        nd->priv = pdat;
    }
}

void DequantizeLinear_reshape(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_type_t otype = b->type;
    y->type = otype;
    tensor_reshape(y, a->ndim, a->dims);
    return;
}

void DequantizeLinear_forward(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch(nd->in[0]->type) {
        case TENSOR_TYPE_INT8:      DequantizeLinear_forward_int8(nd); break;
        default: break;
    }
}

void DequantizeLinear_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_DequantizeLinear_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = DequantizeLinear_init;
    nd->op->reshape     = DequantizeLinear_reshape;
    nd->op->forward     = DequantizeLinear_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = DequantizeLinear_exit;
}
