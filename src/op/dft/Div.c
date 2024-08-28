#include <evo/resolver.h>
#include <evo/util/math.h>

static void Div_int8(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    int8_t* py = (int8_t*)y->datas;
    int8_t* pa;
    int8_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_int16(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    int16_t* py = (int16_t*)y->datas;
    int16_t* pa;
    int16_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_int32(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    int32_t* py = (int32_t*)y->datas;
    int32_t* pa;
    int32_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_int64(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    int64_t* py = (int64_t*)y->datas;
    int64_t* pa;
    int64_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_uint8(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    uint8_t* py = (uint8_t*)y->datas;
    uint8_t* pa;
    uint8_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_uint16(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    uint16_t* py = (uint16_t*)y->datas;
    uint16_t* pa;
    uint16_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_uint32(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    uint32_t* py = (uint32_t*)y->datas;
    uint32_t* pa;
    uint32_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_uint64(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    uint64_t* py = (uint64_t*)y->datas;
    uint64_t* pa;
    uint64_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_float16(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    uint16_t* py = (uint16_t*)y->datas;
    uint16_t* pa;
    uint16_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = float32_to_float16(float16_to_float32(*pa) / float16_to_float32(*pb));
    }
}

static void Div_float32(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    float* py = (float*)y->datas;
    float* pa;
    float* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_float64(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    double* py = (double*)y->datas;
    double* pa;
    double* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = *pa / *pb;
    }
}

static void Div_bfloat16(node_t* nd) {
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    uint16_t* py = (uint16_t*)y->datas;
    uint16_t* pa;
    uint16_t* pb;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        pa = tensor_broadcast_map_address(a, y, i);
        pb = tensor_broadcast_map_address(b, y, i);
        py[i] = float32_to_bfloat16(bfloat16_to_float32(*pa) / bfloat16_to_float32(*pb));
    }
}

void op_Div_dft(node_t* nd) {
    // 1. Div init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    // 2. Div reshape
    tensor_t* y = nd->out[0];
    tensor_t* a = nd->in[0];
    tensor_t* b = nd->in[1];
    tensor_reshape_multi_broadcast(y, a, b, a->type);
    // 3. Div run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT8:
            Div_int8(nd);
            break;
        case TENSOR_TYPE_INT16:
            Div_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Div_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Div_int64(nd);
            break;
        case TENSOR_TYPE_UINT8:
            Div_uint8(nd);
            break;
        case TENSOR_TYPE_UINT16:
            Div_uint16(nd);
            break;
        case TENSOR_TYPE_UINT32:
            Div_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            Div_uint64(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Div_float16(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Div_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Div_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Div_float64(nd);
            break;
        default:
            break;
    }
    // 4. Div exit
    return;
}