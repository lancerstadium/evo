#include <evo/resolver.h>
#include <evo/util/math.h>

typedef struct {
    int m;
    int n;
    int k;
} operator_pdata_t;

static void MatMul_forward_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int32_t *py = (int32_t *)y->datas;
    int32_t *pa;
    int32_t *pb;
    int32_t sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
                py[i + u * pdat->n + v] = sum;
            }
        }
    }
}

static void MatMul_forward_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int64_t *py = (int64_t *)y->datas;
    int64_t *pa;
    int64_t *pb;
    int64_t sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
                py[i + u * pdat->n + v] = sum;
            }
        }
    }
}

static void MatMul_forward_uint32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint32_t *py = (uint32_t *)y->datas;
    uint32_t *pa;
    uint32_t *pb;
    uint32_t sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
                py[i + u * pdat->n + v] = sum;
            }
        }
    }
}

static void MatMul_forward_uint64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint64_t *py = (uint64_t *)y->datas;
    uint64_t *pa;
    uint64_t *pb;
    uint64_t sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
                py[i + u * pdat->n + v] = sum;
            }
        }
    }
}

static void MatMul_forward_bfloat16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa;
    uint16_t *pb;
    float sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += bfloat16_to_float32(pa[u * pdat->k + w]) * bfloat16_to_float32(pb[w * pdat->n + v]);
                py[i + u * pdat->n + v] = float32_to_bfloat16(sum);
            }
        }
    }
}

static void MatMul_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa;
    uint16_t *pb;
    float sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += float16_to_float32(pa[u * pdat->k + w]) * float16_to_float32(pb[w * pdat->n + v]);
                py[i + u * pdat->n + v] = float32_to_float16(sum);
            }
        }
    }
}

static void MatMul_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    float *py = (float *)y->datas;
    float *pa;
    float *pb;
    float sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
                py[i + u * pdat->n + v] = sum;
            }
        }
    }
}

static void MatMul_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    double *py = (double *)y->datas;
    double *pa;
    double *pb;
    double sum;

    for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        for (int u = 0; u < pdat->m; u++) {
            for (int v = 0; v < pdat->n; v++) {
                sum = 0;
                for (int w = 0; w < pdat->k; w++)
                    sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
                py[i + u * pdat->n + v] = sum;
            }
        }
    }
}

void MatMul_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->m = 0;
        pdat->n = 0;
        pdat->k = 0;
        nd->priv = pdat;
    }
}

void MatMul_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int andim;
    int *adims;
    int bndim;
    int *bdims;
    if (a->ndim == 1) {
        adims = (int[]){1, a->dims[0]};
        andim = 2;
    } else {
        adims = a->dims;
        andim = a->ndim;
    }
    if (b->ndim == 1) {
        bdims = (int[]){b->dims[0], 1};
        bndim = 2;
    } else {
        bdims = b->dims;
        bndim = b->ndim;
    }
    int ndim = MAX(andim, bndim);
    int dims[ndim];
    if (andim < 2 || bndim < 2)
        return;
    if (adims[andim - 1] != bdims[bndim - 2])
        return;
    dims[ndim - 2] = adims[andim - 2];
    dims[ndim - 1] = bdims[bndim - 1];
    for (int i = 3; i <= ndim; i++) {
        int alen = (andim - i) < 0 ? 1 : adims[andim - i];
        int blen = (bndim - i) < 0 ? 1 : bdims[bndim - i];
        if (alen != blen && alen > 1 && blen > 1)
            return;
        dims[ndim - i] = MAX(alen, blen);
    }
    pdat->m = adims[andim - 2];
    pdat->n = bdims[bndim - 1];
    pdat->k = adims[andim - 1];
    y->type = a->type;
    tensor_reshape(y, ndim, dims);
}

void MatMul_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch(nd->in[0]->type) {
        case TENSOR_TYPE_INT32:
            MatMul_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            MatMul_forward_int64(nd);
            break;
        case TENSOR_TYPE_UINT32:
            MatMul_forward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            MatMul_forward_uint64(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            MatMul_forward_float16(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            MatMul_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            MatMul_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            MatMul_forward_float64(nd);
            break;
        default:
            break;
    }
}

void MatMul_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}


void op_MatMul_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = MatMul_init;
    nd->op->reshape     = MatMul_reshape;
    nd->op->forward     = MatMul_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = MatMul_exit;
}