#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>


typedef struct {
    char* mode;
} operator_pdata_t;

void Pad_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->mode = node_get_attr_string(nd, "mode", "constant");
        nd->priv = pdat;
    }
}

#include <evo/util/log.h>

// ref: https://onnx.ai/onnx/operators/onnx__Pad.html
void Pad_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_t* y = nd->out[0];
    LOG_INFO("%08u", shash(pdat->mode));
    switch(shash(pdat->mode)) {
        case 0: {   /* constant */
            int64_t constant = 0;
            if(nd->nin >= 3 && nd->in[2]->type == a->type) {
                int64_t* cd = nd->in[2]->datas;
                constant = cd[0];
            }
            break;
        }
        default: break;
    }
}

void Pad_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Pad_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Pad_init;
    nd->op->reshape     = Pad_reshape;
    nd->op->forward     = NULL;
    nd->op->backward    = NULL;
    nd->op->exit        = Pad_exit;
}