#include <evo/resolver.h>
#include <evo/util/log.h>
#include <string.h>

typedef struct {
    int axis;
} operator_pdata_t;

void Gather_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->axis = node_get_attr_int(nd, "axis", 0);
        nd->priv = pdat;
    }
}

void Gather_reshape(node_t* nd) {
    //   in[0]      in[1]     axis         out[0]
    // [3,4,3,4]    [2,3]       0   ->  [2,3,4,3,4]
    // [3,4,3,4]    [2,3]       1   ->  [3,2,3,3,4]
    // [3,4,3,4]    [2,3]       2   ->  [3,4,2,3,4]
    // [3,4,3,4]    [2,3]       3   ->  [3,4,3,2,3]
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_t* c = nd->out[0];
    int ndim = a->ndim + b->ndim - 1;
    int dims[ndim];
    int axis = pdat->axis;
    if (axis < 0)
        axis += a->ndim;
    if (axis < 0 || axis >= a->ndim)
        axis = 0;
    int i = 0, j = 0;
    while(i < ndim) {
        if(i == axis) {
            for(int k = 0; k < b->ndim; k++) {
                dims[i] = b->dims[k];
                i++;
            }
            j++;
        } else {
            dims[i] = a->dims[j];
            i++; j++;
        }
    }
    c->type = a->type;
    tensor_reshape(c, ndim, dims);
}

void Gather_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_t* c = nd->out[0];
    int axis = pdat->axis;
    int blocks =  1;
    for(int i = 0; i < axis; i++) {
        blocks *= a->dims[i];
    }
    int step = a->strides[axis] * tensor_type_sizeof(a->type);
    int step2 = axis > 0 ? a->strides[axis - 1] * tensor_type_sizeof(a->type) : 0;
    if(b->type == TENSOR_TYPE_INT32) {
        int32_t* idxs = b->datas;
        for(int j = 0; j < blocks; j++) {
            for(int i = 0; i < b->ndata; i++) {
                if(idxs[i] < 0 || idxs[i] >= a->dims[axis]) {
                    memset(c->datas + (j * b->ndata + i) * step, 0, step);
                } else {
                    memcpy(c->datas + (j * b->ndata + i) * step,
                     a->datas + j * step2 + idxs[i] * step, 
                     step);
                }
            }
        }
    } else if(b->type == TENSOR_TYPE_INT64) {
        int64_t* idxs = b->datas;
        for(int j = 0; j < blocks; j++) {
            for(int i = 0; i < b->ndata; i++) {
                if(idxs[i] < 0 || idxs[i] >= a->dims[axis]) {
                    memset(c->datas + (j * b->ndata + i) * step, 0, step);
                } else {
                    memcpy(c->datas + (j * b->ndata + i) * step,
                     a->datas + j * step2 + idxs[i] * step, 
                     step);
                }
            }
        }
    } else {
        LOG_ERR("Gather: unexcept index tensor type!\n");
    }
}

void Gather_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Gather_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Gather_init;
    nd->op->reshape     = Gather_reshape;
    nd->op->forward     = Gather_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Gather_exit;
}