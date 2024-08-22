#include "../../evo/resolver.h"
#include "../../util/math.h"

static void Shape_operator(node_t* nd) {
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    int64_t* py = (int64_t*)y->datas;
    size_t i, l;

    for (i = 0, l = MIN(y->ndata, (size_t)x->ndim); i < l; i++)
        py[i] = x->dims[i];
}

void op_Shape_dft(node_t* nd) {
    // 1. Shape init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    // 2. Shape reshape
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    y->type = TENSOR_TYPE_INT64;
    tensor_reshape(y, 1, (int[]){x->ndim});
    // 3. Shape run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BOOL:
        case TENSOR_TYPE_INT8:
        case TENSOR_TYPE_INT16:
        case TENSOR_TYPE_INT32:
        case TENSOR_TYPE_INT64:
        case TENSOR_TYPE_UINT8:
        case TENSOR_TYPE_UINT16:
        case TENSOR_TYPE_UINT32:
        case TENSOR_TYPE_UINT64:
        case TENSOR_TYPE_FLOAT16:
        case TENSOR_TYPE_FLOAT32:
        case TENSOR_TYPE_FLOAT64:
        case TENSOR_TYPE_COMPLEX64:
        case TENSOR_TYPE_COMPLEX128:
        case TENSOR_TYPE_STRING:
            Shape_operator(nd);
            break;
        default:
            break;
    }
    // 4. Shape exit
    return;
}