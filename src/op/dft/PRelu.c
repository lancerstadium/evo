#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>


static void PRelu_forward_int32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int32_t *py = (int32_t *)y->datas;
    int32_t *pa = (int32_t *)a->datas;;
    int32_t *pb = (int32_t *)b->datas;;
    PRelu_forward_int32_cpu(pa, pb, py, a->ndata);
}
static void PRelu_forward_int64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int64_t *py = (int64_t *)y->datas;
    int64_t *pa = (int64_t *)a->datas;
    int64_t *pb = (int64_t *)b->datas;
    PRelu_forward_int64_cpu(pa, pb, py, a->ndata);
}
static void PRelu_forward_uint32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint32_t *py = (uint32_t *)y->datas;
    uint32_t *pa = (uint32_t *)a->datas;
    uint32_t *pb = (uint32_t *)b->datas;
    PRelu_forward_uint32_cpu(pa, pb, py, a->ndata);
}
static void PRelu_forward_uint64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint64_t *py = (uint64_t *)y->datas;
    uint64_t *pa = (uint64_t *)a->datas;
    uint64_t *pb = (uint64_t *)b->datas;
    PRelu_forward_uint64_cpu(pa, pb, py, a->ndata);
}
static void PRelu_forward_float32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    float *py = (float *)y->datas;
    float *pa = (float *)a->datas;
    float *pb = (float *)b->datas;
    PRelu_forward_float32_cpu(pa, pb, py, a->ndata);
}
static void PRelu_forward_float64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    double *py = (double *)y->datas;
    double *pa = (double *)a->datas;
    double *pb = (double *)b->datas;
    PRelu_forward_float64_cpu(pa, pb, py, a->ndata);
}



static void PRelu_backward_float32(node_t *nd) {
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *da = nd->in[0]->grad;
    tensor_t *db = nd->in[0]->grad;
    tensor_t *dy = nd->out[0]->grad;
    float *pa = (float *)a->datas;
    float *pb = (float *)b->datas;
    float *pda = (float *)da->datas;
    float *pdb = (float *)db->datas;
    float *pdy = (float *)dy->datas;
    PRelu_backward_float32_cpu(pa, pb, pdy, pda, pdb, dy->ndata);
}
static void PRelu_backward_float64(node_t *nd) {
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *da = nd->in[0]->grad;
    tensor_t *db = nd->in[0]->grad;
    tensor_t *dy = nd->out[0]->grad;
    double *pa = (double *)a->datas;
    double *pb = (double *)b->datas;
    double *pda = (double *)da->datas;
    double *pdb = (double *)db->datas;
    double *pdy = (double *)dy->datas;
    PRelu_backward_float64_cpu(pa, pb, pdy, pda, pdb, dy->ndata);
}

void PRelu_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void PRelu_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_reshape_multi_broadcast(y, a, b, a->type);
}

void PRelu_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT32:     PRelu_forward_int32(nd); break;
        case TENSOR_TYPE_INT64:     PRelu_forward_int64(nd); break;
        case TENSOR_TYPE_UINT32:    PRelu_forward_uint32(nd); break;
        case TENSOR_TYPE_UINT64:    PRelu_forward_uint64(nd); break;
        case TENSOR_TYPE_FLOAT32:   PRelu_forward_float32(nd); break;
        case TENSOR_TYPE_FLOAT64:   PRelu_forward_float64(nd); break;
        default: break;
    }
}

void PRelu_backward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->in[0]->name);
        nd->in[0]->grad = tensor_new(name_buf, nd->in[0]->type);
        tensor_reshape(nd->in[0]->grad, nd->in[0]->ndim, nd->in[0]->dims);
    }
    switch (nd->in[0]->type) {
        // case TENSOR_TYPE_INT32:     PRelu_backward_int32(nd); break;
        // case TENSOR_TYPE_INT64:     PRelu_backward_int64(nd); break;
        // case TENSOR_TYPE_UINT32:    PRelu_backward_uint32(nd); break;
        // case TENSOR_TYPE_UINT64:    PRelu_backward_uint64(nd); break;
        case TENSOR_TYPE_FLOAT32:   PRelu_backward_float32(nd); break;
        case TENSOR_TYPE_FLOAT64:   PRelu_backward_float64(nd); break;
        default: break;
    }
}

void PRelu_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_PRelu_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = PRelu_init;
    nd->op->reshape     = PRelu_reshape;
    nd->op->forward     = PRelu_forward;
    nd->op->backward    = PRelu_backward;
    nd->op->exit        = PRelu_exit;
}