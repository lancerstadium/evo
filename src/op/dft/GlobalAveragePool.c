#include <string.h>

#include <evo/resolver.h>
#include <evo/util/math.h>

static void GlobalAveragePool_forward_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    int N = y->dims[0];
    int C = y->dims[1];
    int avgsz = x->ndata / (N * C);
    float sum[N][C];
    int idx[2], cnt;
    size_t i, j, l;

    memset(sum, 0, sizeof(sum));
    for (i = 0, l = x->ndata; i < l; i++) {
        cnt = i;
        idx[0] = cnt / x->strides[0];
        cnt %= x->strides[0];
        idx[1] = cnt / x->strides[1];
        cnt %= x->strides[1];
        sum[idx[0]][idx[1]] += float16_to_float32(px[i]);
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < C; j++)
            py[i * C + j] = float32_to_float16(sum[i][j] / avgsz);
    }
}

static void GlobalAveragePool_forward_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    int N = y->dims[0];
    int C = y->dims[1];
    int avgsz = x->ndata / (N * C);
    float sum[N][C];
    int idx[2], cnt;
    size_t i, j, l;

    memset(sum, 0, sizeof(sum));
    for (i = 0, l = x->ndata; i < l; i++) {
        cnt = i;
        idx[0] = cnt / x->strides[0];
        cnt %= x->strides[0];
        idx[1] = cnt / x->strides[1];
        cnt %= x->strides[1];
        sum[idx[0]][idx[1]] += px[i];
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < C; j++)
            py[i * C + j] = sum[i][j] / avgsz;
    }
}

static void GlobalAveragePool_forward_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    int N = y->dims[0];
    int C = y->dims[1];
    int avgsz = x->ndata / (N * C);
    double sum[N][C];
    int idx[2], cnt;
    size_t i, j, l;

    memset(sum, 0, sizeof(sum));
    for (i = 0, l = x->ndata; i < l; i++) {
        cnt = i;
        idx[0] = cnt / x->strides[0];
        cnt %= x->strides[0];
        idx[1] = cnt / x->strides[1];
        cnt %= x->strides[1];
        sum[idx[0]][idx[1]] += px[i];
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < C; j++)
            py[i * C + j] = sum[i][j] / avgsz;
    }
}

void GlobalAveragePool_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
}

void GlobalAveragePool_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int ndim = x->ndim;
    int dims[ndim];
    int i;
    for (i = 0; i < ndim; i++) {
        if (i < 2)
            dims[i] = x->dims[i];
        else
            dims[i] = 1;
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
}

void GlobalAveragePool_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            GlobalAveragePool_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            GlobalAveragePool_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            GlobalAveragePool_forward_float64(nd);
            break;
        default:
            break;
    }
}

void GlobalAveragePool_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_GlobalAveragePool_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = GlobalAveragePool_init;
    nd->op->reshape     = GlobalAveragePool_reshape;
    nd->op->forward     = GlobalAveragePool_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = GlobalAveragePool_exit;
}