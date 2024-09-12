#include <evo/resolver.h>
#include <evo/util/sys.h>
#include <string.h>

static void Where_bool(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint8_t *py = (uint8_t *)y->datas;
    uint8_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_int8(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    int8_t *py = (int8_t *)y->datas;
    int8_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_int16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    int16_t *py = (int16_t *)y->datas;
    int16_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_int32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    int32_t *py = (int32_t *)y->datas;
    int32_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_int64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    int64_t *py = (int64_t *)y->datas;
    int64_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_uint8(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint8_t *py = (uint8_t *)y->datas;
    uint8_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_uint16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_uint32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint32_t *py = (uint32_t *)y->datas;
    uint32_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_uint64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint64_t *py = (uint64_t *)y->datas;
    uint64_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_bfloat16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_float16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_float32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    float *py = (float *)y->datas;
    float *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_float64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    double *py = (double *)y->datas;
    double *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i] = *px;
    }
}

static void Where_complex64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    float *py = (float *)y->datas;
    float *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i * 2] = px[0];
        py[i * 2 + 1] = px[1];
    }
}

static void Where_complex128(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    double *py = (double *)y->datas;
    double *px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = tensor_broadcast_map_address(x1, y, i);
        else
            px = tensor_broadcast_map_address(x2, y, i);
        py[i * 2] = px[0];
        py[i * 2 + 1] = px[1];
    }
}

static void Where_string(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *x0 = nd->in[0];
    tensor_t *x1 = nd->in[1];
    tensor_t *x2 = nd->in[2];
    char **py = (char **)y->datas;
    char **px;
    uint8_t *c;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        c = tensor_broadcast_map_address(x0, y, i);
        if (*c)
            px = (char **)tensor_broadcast_map_address(x1, y, i);
        else
            px = (char **)tensor_broadcast_map_address(x2, y, i);
        if (py[i])
            free(py[i]);
        py[i] = strdup(px[i]);
    }
}

void Where_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}


void Where_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t *y = nd->out[0];
    int i;
    if (!tensor_reshape_ident(y, nd->in[nd->nin - 1], nd->in[nd->nin - 1]->type))
        return;
    for (i = nd->nin - 2; i >= 0; i--) {
        if (!tensor_reshape_multi_broadcast(y, y, nd->in[i], y->type))
            return;
    }
}

void Where_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 3) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) || (nd->in[2]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED || nd->in[2]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BOOL:
            Where_bool(nd);
            break;
        case TENSOR_TYPE_INT8:
            Where_int8(nd);
            break;
        case TENSOR_TYPE_INT16:
            Where_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Where_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Where_int64(nd);
            break;
        case TENSOR_TYPE_UINT8:
            Where_uint8(nd);
            break;
        case TENSOR_TYPE_UINT16:
            Where_uint16(nd);
            break;
        case TENSOR_TYPE_UINT32:
            Where_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            Where_uint64(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Where_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Where_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Where_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Where_float64(nd);
            break;
        case TENSOR_TYPE_STRING:
            Where_string(nd);
            break;
        case TENSOR_TYPE_COMPLEX64:
            Where_complex64(nd);
            break;
        case TENSOR_TYPE_COMPLEX128:
            Where_complex128(nd);
            break;
        default:
            break;
    }
}

void Where_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}


void op_Where_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Where_init;
    nd->op->reshape     = Where_reshape;
    nd->op->forward     = Where_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Where_exit;
}