#include "../../evo/resolver.h"
#include "../../util/sys.h"
#include <string.h>

typedef struct {
    int axis;
} operator_pdata_t;

static void Flatten_operator(node_t* nd) {
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    char** px = (char**)x->datas;
    char** py = (char**)y->datas;

    if (x->type == TENSOR_TYPE_STRING) {
        for (size_t i = 0, l = y->ndata; i < l; i++) {
            if (py[i])
                free(py[i]);
            py[i] = sys_strdup(px[i]);
        }
    } else {
        memcpy(y->datas, x->datas, x->ndata * tensor_type_sizeof(x->type));
    }
}

void op_Flatten_dft(node_t* nd) {
    // 1. Flatten init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->axis = node_get_attr_int(nd, "axis", 1);
        nd->priv = pdat;
    }
    // 2. Flatten reshape
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    int axis = pdat->axis;
    int dims[x->ndim];
    int ndim;
    int i, j;

    if (axis < 0)
        axis += x->ndim;
    if (axis < 0 || axis >= x->ndim)
        return;
    for (i = 0, j = 1, ndim = 0; i < x->ndim; i++) {
        if (i != axis)
            j *= x->dims[i];
        else {
            dims[ndim++] = j;
            j = x->dims[i];
        }
    }
    dims[ndim++] = j;
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
    // 3. Flatten run
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
        case TENSOR_TYPE_BFLOAT16:
        case TENSOR_TYPE_FLOAT16:
        case TENSOR_TYPE_FLOAT32:
        case TENSOR_TYPE_FLOAT64:
        case TENSOR_TYPE_COMPLEX64:
        case TENSOR_TYPE_COMPLEX128:
        case TENSOR_TYPE_STRING:
            Flatten_operator(nd);
            break;
        default:
            break;
    }
    // 4. Flatten exit
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}