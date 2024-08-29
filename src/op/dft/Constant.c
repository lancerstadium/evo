#include <string.h>

#include <evo/resolver.h>

void op_Constant_dft(node_t *nd) {
    // 1. Constant init
    if (!(nd->nout == 1) || !(vector_size(nd->attr_vec) == 1)) {
        return;
    }
    tensor_t *y = nd->out[0];
    attribute_t *attr = nd->attr_vec[0];
    if (attr) {
        switch (attr->type) {
            case ATTRIBUTE_TYPE_FLOAT:
                if (strcmp(attr->name, "value_float") == 0) {
                    if ((y->ndim != 0) || (y->type != TENSOR_TYPE_FLOAT32))
                        tensor_reinit(y, TENSOR_TYPE_FLOAT32, 0, NULL);
                    tensor_apply(y, &attr->f, sizeof(float));
                }
                break;
            case ATTRIBUTE_TYPE_INT:
                if (strcmp(attr->name, "value_int") == 0) {
                    if ((y->ndim != 0) || (y->type != TENSOR_TYPE_INT64))
                        tensor_reinit(y, TENSOR_TYPE_INT64, 0, NULL);
                    tensor_apply(y, &attr->i, sizeof(int64_t));
                }
                break;
            case ATTRIBUTE_TYPE_STRING:
                break;
            case ATTRIBUTE_TYPE_FLOATS:
                if ((strcmp(attr->name, "value_floats") == 0) && (attr->nf > 0)) {
                    if ((y->ndim != 1) || (y->dims[0] != attr->nf) || (y->type != TENSOR_TYPE_FLOAT32))
                        tensor_reinit(y, TENSOR_TYPE_FLOAT32, 1, (int[]){attr->nf});
                    tensor_apply(y, attr->fs, attr->nf * sizeof(float));
                }
                break;
            case ATTRIBUTE_TYPE_INTS:
                if ((strcmp(attr->name, "value_ints") == 0) && (attr->ni > 0)) {
                    if ((y->ndim != 1) || (y->dims[0] != attr->ni) || (y->type != TENSOR_TYPE_INT64))
                        tensor_reinit(y, TENSOR_TYPE_INT64, 1, (int[]){attr->ni});
                    tensor_apply(y, attr->is, attr->ni * sizeof(int64_t));
                }
                break;
            // case ATTRIBUTE_TYPE_STRINGS:
            case ATTRIBUTE_TYPE_TENSOR:
                y = node_get_attr_tensor(nd, "value", y);
                break;
            default:
                break;
        }
    }
    // 2. Constant reshape
    // 3. Constant run
    
    // 4. Constant exit
    return;
}