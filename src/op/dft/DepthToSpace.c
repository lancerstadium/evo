#include <evo/resolver.h>
#include <string.h>

typedef struct {
    int64_t blocksize;
    char* mode;
} operator_pdata_t;

void DepthToSpace_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if(pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->blocksize = node_get_attr_int(nd, "blocksize", 1);
        pdat->mode = node_get_attr_string(nd, "mode", "DCR");
        nd->priv = pdat;
    }
}

void DepthToSpace_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->ndim != 4
        || nd->in[0]->layout == 1) {
        return;
    }
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    operator_pdata_t* pdat = nd->priv;
    y->type = x->type;
    tensor_reshape(y, 4, (int[]){x->dims[0], x->dims[1] / (int)(pdat->blocksize * pdat->blocksize), x->dims[2] * (int)pdat->blocksize, x->dims[3] * (int)pdat->blocksize});
}

void DepthToSpace_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->ndim != 4
        || nd->in[0]->layout == 1) {
        return;
    }
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    void* px = x->datas;
    void* py = y->datas;
    int sz = tensor_type_sizeof(x->type);
    operator_pdata_t* pdat = nd->priv;
    // blocksize == 1
    if(pdat->blocksize == 1 && px && py) {
        memcpy(py, px, x->ndata * sz);
        return;
    }
    // other
    int H = x->dims[2];     // Height
    int W = x->dims[3];     // Width
    int bsz = pdat->blocksize;
    int step1 = y->dims[1];                         // Step1: for small block
    int step2 = x->dims[1];                         // Step2: for middle block, Step3 = 1
    int stride1 = y->strides[1];
    for(int i = 0; i < step1; i++) {                // Big Iter: New Channel
        for(int j = 0; j < W; j++) {                // W
            for(int k = 0; k < H; k++) {            // H
                for(int l = 0; l < bsz; l++) {      // W
                    for(int m = 0; m < bsz; m++) {  // H
                        memcpy(py + (i * stride1 + ((k * W * bsz + m) + j) * bsz + l) * sz, px + (i + (k * W + j) * step2 + (m * bsz + l) * step1) * sz, sz);
                    }
                }
            }
        }
    }
    return;
}

void DepthToSpace_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_DepthToSpace_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = DepthToSpace_init;
    nd->op->reshape     = DepthToSpace_reshape;
    nd->op->forward     = DepthToSpace_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = DepthToSpace_exit;
}