#include <evo/resolver.h>
#include <evo/util/math.h>

static void Add_forward_int8(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int8_t *py = (int8_t *)y->datas;
    int8_t *pa;
    int8_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_int16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int16_t *py = (int16_t *)y->datas;
    int16_t *pa;
    int16_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_int32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int32_t *py = (int32_t *)y->datas;
    int32_t *pa;
    int32_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_int64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int64_t *py = (int64_t *)y->datas;
    int64_t *pa;
    int64_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_uint8(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint8_t *py = (uint8_t *)y->datas;
    uint8_t *pa;
    uint8_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_uint16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa;
    uint16_t *pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_uint32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint32_t *py = (uint32_t *)y->datas;
    uint32_t *pa;
    uint32_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_uint64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint64_t *py = (uint64_t *)y->datas;
    uint64_t *pa;
    uint64_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_float16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa;
    uint16_t *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = float32_to_float16(float16_to_float32(*pa) + float16_to_float32(*pb));
    }
}

static void Add_forward_float32(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    float *py = (float *)y->datas;
    float *pa;
    float *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_float64(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    double *py = (double *)y->datas;
    double *pa;
    double *pb;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa + *pb;
    }
}

static void Add_forward_bfloat16(node_t *nd) {
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa;
    uint16_t *pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = float32_to_bfloat16(bfloat16_to_float32(*pa) + bfloat16_to_float32(*pb));
    }
}

void Add_forward_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
}

void Add_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_reshape_multi_broadcast(y, a, b, a->type);
}

void Add_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT8:
            Add_forward_int8(nd);
            break;
        case TENSOR_TYPE_INT16:
            Add_forward_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Add_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Add_forward_int64(nd);
            break;
        case TENSOR_TYPE_UINT8:
            Add_forward_uint8(nd);
            break;
        case TENSOR_TYPE_UINT16:
            Add_forward_uint16(nd);
            break;
        case TENSOR_TYPE_UINT32:
            Add_forward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            Add_forward_uint64(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Add_forward_float16(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Add_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Add_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Add_forward_float64(nd);
            break;
        default:
            break;
    }
}

void Add_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Add_dft(node_t *nd) {
    // 1. Add init
    Add_forward_init(nd);
    // 2. Add reshape
    Add_reshape(nd);
    // 3. Add forward
    Add_forward(nd);
    // 4. Add exit
    Add_exit(nd);
}