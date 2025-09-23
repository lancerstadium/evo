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
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->reduction = node_get_attr_string(nd, "reduction", "none");
        nd->priv = pdat;
    }
}


void ScatterNd_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* a = nd->in[0];    /* data */
    tensor_t* y = nd->out[0];   /* outs */
    y->type = a->type;
    tensor_reshape(y, a->ndim, a->dims);
}

void ScatterNd_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if(nd->nin < 3 || nd->in[1]->type != TENSOR_TYPE_INT64) return;
    tensor_t* a = nd->in[0];    /* dats */
    tensor_t* b = nd->in[1];    /* inds */
    tensor_t* c = nd->in[2];    /* upds */
    tensor_t* y = nd->out[0];   /* outs */

    int ndimd = b->dims[b->ndim - 1];       /* data dims */
    int ndimi = b->ndim - 1;                /* indx dims */
    tensor_apply(y, a->datas, a->ndata * tensor_type_sizeof(a->type));
    if(ndimd > a->ndim || c->ndim != (a->ndim - ndimd + ndimi)) return;
    int stepa = a->strides[ndimd - 1] * tensor_type_sizeof(a->type);
    int stepb = b->dims[b->ndim - 1] * tensor_type_sizeof(b->type);
    int stepc = c->strides[ndimi  - 1] * tensor_type_sizeof(c->type);

    int ninds = 1;  /* indices 前几维的数量 */
    for (int i = 0; i < ndimi; i++) {
        ninds *= c->dims[i];  /* 计算索引的数量 */
    }

    /* 遍历所有的 indices */
    for (int i = 0; i < ninds; i++) {
        /* 获取 indices 中第 i 组索引 */
        int64_t* idx = (int64_t *)(b->datas + i * stepb);

        /* 计算 data 中对应的位置 */
        int offset = 0;
        for (int j = 0; j < ndimd; j++) {
            if (idx[j] < 0 || idx[j] >= a->dims[j]) {
                /* 索引超出范围，跳过这次更新 */
                return;
            }
            offset += idx[j] * a->strides[j] * tensor_type_sizeof(a->type);
        }

        /* 检查边界：确保 offset 在 y 张量的范围内 */
        if (offset + stepa > y->ndata * tensor_type_sizeof(y->type)) {
            /* 如果目标位置超出 y 的尺寸，跳过该次更新 */
            return;
        }

        /* 将 updates 的值写入到 y 中相应位置 */
        void* src = (void *)(c->datas + i * stepc);  /* 更新值 */
        void* dst = (void *)(y->datas + offset);     /* 目标位置 */
        
        /* 进行更新，确保多次赋值 */
        memcpy(dst, src, stepa);  /* 将 updates 中的值复制到 y 张量中 */
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