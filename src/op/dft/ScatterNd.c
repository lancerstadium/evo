#include <evo/resolver.h>
#include <evo/util/log.h>
#include <string.h>

typedef struct {
    char* reduction;
} operator_pdata_t;

void ScatterNd_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->reduction = node_get_attr_string(nd, "reduction", "none");
        nd->priv = pdat;
    }
}


void ScatterNd_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t* a = nd->in[0];    /* data */
    tensor_t* y = nd->out[0];   /* outs */
    y->type = a->type;
    tensor_reshape(y, a->ndim, a->dims);
}

void ScatterNd_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(nd->nin < 3 || nd->in[1]->type != TENSOR_TYPE_INT64) return;
    tensor_t* a = nd->in[0];    /* dats */
    tensor_t* b = nd->in[1];    /* inds */
    tensor_t* c = nd->in[2];    /* upds */
    tensor_t* y = nd->out[0];   /* outs */
    int64_t* inds = b->datas;
    int step1 = a->strides[0] * tensor_type_sizeof(a->type);
    int step2 = c->strides[0] * tensor_type_sizeof(a->type);
    if(step1 <= 0 || step2 <= 0) return;
    int times = step1 / step2;
    int min_step = step1 > step2 ? step2 : step1;
    LOG_INFO("%d, %d, %d, %d\n", step1, step2, times, min_step);
    tensor_apply(y, a->datas, a->ndata * tensor_type_sizeof(a->type));
    for(int i = 0; i < b->ndata; i++) {
        if(inds[i] >= a->dims[0]) continue;
        for(int j = 0; j < times; j++) {
            memcpy(y->datas + step1 * inds[i] + min_step * j, c->datas + step2 * i, min_step);
        }
    }
}

void ScatterNd_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_ScatterNd_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = ScatterNd_init;
    nd->op->reshape     = ScatterNd_reshape;
    nd->op->forward     = ScatterNd_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = ScatterNd_exit;
}