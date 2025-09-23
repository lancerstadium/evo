#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>

typedef struct {
    int64_t axis;               // 1
    int64_t block_size;         // 0
    int64_t output_dtype;       // 0
    int64_t saturate;           // 1
} operator_pdata_t;


static void QuantizeLinear_forward_float32(node_t* nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_t* c = nd->nin >= 3 ? nd->in[2] : NULL;
    float* sc = b->datas;
    switch(y->type) {
        case TENSOR_TYPE_UINT8: {
            if(b->ndata == 1) {     // Per-tensor Quant
                Quant_asymmetric_float32_to_uint8_cpu(a->datas, y->datas, a->ndata, sc[0], (c ? ((uint8_t*)(c->datas))[0] : 0));
            }
            break;
        }
        case TENSOR_TYPE_INT8: {
            if(b->ndata == 1) {     // Per-tensor Quant
                Quant_asymmetric_float32_to_int8_cpu(a->datas, y->datas, a->ndata, sc[0], (c ? ((int8_t*)(c->datas))[0] : 0));
            }
            break;
        }
        case TENSOR_TYPE_INT32: {
            if(b->ndata == 1) {     // Per-tensor Quant
                Quant_asymmetric_float32_to_int32_cpu(a->datas, y->datas, a->ndata, sc[0], (c ? ((int32_t*)(c->datas))[0] : 0));
            }
            break;
        }
        default: break;
    }
}


void QuantizeLinear_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->axis = node_get_attr_int(nd, "axis", 1);
        pdat->block_size = node_get_attr_int(nd, "block_size", 0);
        pdat->output_dtype = node_get_attr_int(nd, "output_dtype", (int64_t)TENSOR_TYPE_UINT8);
        pdat->saturate = node_get_attr_int(nd, "saturate", 1);
        nd->priv = pdat;
    }
}

void QuantizeLinear_reshape(node_t* nd) {
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
    tensor_t* c = nd->nin >= 3 ? nd->in[2] : NULL;
    tensor_type_t otype = c ? c->type : (tensor_type_t)pdat->output_dtype;
    y->type = otype;
    tensor_reshape(y, a->ndim, a->dims);
    return;
}

void QuantizeLinear_forward(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch(nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT32:   QuantizeLinear_forward_float32(nd); break;
        default: break;
    }
}

void QuantizeLinear_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_QuantizeLinear_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = QuantizeLinear_init;
    nd->op->reshape     = QuantizeLinear_reshape;
    nd->op->forward     = QuantizeLinear_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = QuantizeLinear_exit;
}
