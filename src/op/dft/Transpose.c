#include <evo/resolver.h>

typedef struct {
    int *perm;
    int nperm;
} operator_pdata_t;

static void Transpose_bool(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    uint8_t *py = (uint8_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_int8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_int16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int16_t *px = (int16_t *)x->datas;
    int16_t *py = (int16_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int64_t *px = (int64_t *)x->datas;
    int64_t *py = (int64_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_uint8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    uint8_t *py = (uint8_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_uint16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_uint32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint32_t *px = (uint32_t *)x->datas;
    uint32_t *py = (uint32_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_uint64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint64_t *px = (uint64_t *)x->datas;
    uint64_t *py = (uint64_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_bfloat16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
    }
}

static void Transpose_forward_complex64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
        py[oy + 1] = px[ox + 1];
    }
}

static void Transpose_forward_complex128(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        py[oy] = px[ox];
        py[oy + 1] = px[ox + 1];
    }
}

static void Transpose_forward_string(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    char **px = (char **)x->datas;
    char **py = (char **)y->datas;
    int nperm = pdat->nperm;
    int ix[nperm], iy[nperm];
    int ox, oy;
    size_t i, l;

    for (oy = 0, l = y->ndata; oy < l; oy++) {
        tensor_offset2index(y, oy, iy);
        for (i = 0; i < nperm; i++)
            ix[pdat->perm[i]] = iy[i];
        ox = tensor_index2offset(x, ix);
        if (py[oy])
            free(py[oy]);
        py[oy] = sys_strdup(px[ox]);
    }
}

void Transpose_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = sys_malloc(sizeof(operator_pdata_t));
    int64_t *ints;
    int i;
    if (pdat) {
        pdat->nperm = nd->in[0]->ndim;
        pdat->perm = sys_malloc(sizeof(int) * pdat->nperm);
        if (pdat->nperm) {
            if (pdat->nperm == node_get_attr_ints(nd, "perm", &ints)) {
                for (i = 0; i < pdat->nperm; i++)
                    pdat->perm[i] = ints[i];
            }
        } else {
            for (i = 0; i < pdat->nperm; i++)
                pdat->perm[i] = pdat->nperm - i - 1;
        }
        nd->priv = pdat;
    } else {
        free(pdat);
        return;
    }
}

void Transpose_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    int i;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    if (tensor_reshape_ident(y, x, x->type)) {
        for (i = 0; i < x->ndim; i++)
            y->dims[i] = x->dims[pdat->perm[i]];
    }
}

void Transpose_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BOOL:
            Transpose_bool(nd);
            break;
        case TENSOR_TYPE_INT8:
            Transpose_forward_int8(nd);
            break;
        case TENSOR_TYPE_INT16:
            Transpose_forward_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Transpose_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Transpose_forward_int64(nd);
            break;
        case TENSOR_TYPE_UINT8:
            Transpose_forward_uint8(nd);
            break;
        case TENSOR_TYPE_UINT16:
            Transpose_forward_uint16(nd);
            break;
        case TENSOR_TYPE_UINT32:
            Transpose_forward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            Transpose_forward_uint64(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Transpose_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Transpose_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Transpose_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Transpose_forward_float64(nd);
            break;
        case TENSOR_TYPE_COMPLEX64:
            Transpose_forward_complex64(nd);
            break;
        case TENSOR_TYPE_COMPLEX128:
            Transpose_forward_complex128(nd);
            break;
        case TENSOR_TYPE_STRING:
            Transpose_forward_string(nd);
            break;
        default:
            break;
    }

}

void Transpose_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat) {
        if (pdat->perm)
            free(pdat->perm);
        free(pdat);
    }
    nd->priv = NULL;
    return;
}

void op_Transpose_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Transpose_init;
    nd->op->reshape     = Transpose_reshape;
    nd->op->forward     = Transpose_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Transpose_exit;
}