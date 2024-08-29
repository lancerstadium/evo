#include <evo/resolver.h>
#include <string.h>

static void Squeeze_operator(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    char **px = (char **)x->datas;
    char **py = (char **)y->datas;
    if (x->type == TENSOR_TYPE_STRING) {
        for (size_t i = 0, l = y->ndata; i < l; i++) {
            if (py[i])
                free(py[i]);
            py[i] = strdup(px[i]);
        }
    } else {
        memcpy(y->datas, x->datas, x->ndata * tensor_type_sizeof(x->type));
    }
}

void op_Squeeze_dft(node_t *nd) {
    // 1. Squeeze init
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    // 2. Squeeze reshape
    tensor_t *y = nd->out[0];
    tensor_t *x = nd->in[0];
    tensor_t *a;
    int64_t *pa;
    int dims[x->ndim];
    int ndim = 0;
    int axis, flag;
    int i, j;
    if (nd->nin > 1) {
        a = nd->in[1];
        pa = (int64_t *)a->datas;
        for (i = 0, ndim = 0; i < x->ndim; i++) {
            if (x->dims[i] > 1)
                dims[ndim++] = x->dims[i];
            else {
                for (j = 0, flag = 0; j < a->ndata; j++) {
                    axis = pa[j];
                    if (axis < 0)
                        axis += x->ndim;
                    if (i == axis) {
                        flag = 1;
                        break;
                    }
                }
                if (!flag)
                    dims[ndim++] = x->dims[i];
            }
        }
    } else {
        for (i = 0, ndim = 0; i < x->ndim; i++) {
            if (x->dims[i] > 1)
                dims[ndim++] = x->dims[i];
        }
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
    // 3. Squeeze run
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
            Squeeze_operator(nd);
            break;
        default:
            break;
    }
    // 4. Squeeze exit
    return;
}