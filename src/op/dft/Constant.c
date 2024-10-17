#include <evo/resolver.h>
#include <string.h>

void Constant_init(node_t *nd) {
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
                tensor_t* ts = node_get_attr_tensor(nd, "value", y);
                y->type = ts->type;
                tensor_reshape(y, ts->ndim, ts->dims);
                tensor_apply(y, ts->datas, ts->ndata * tensor_type_sizeof(ts->type));
                tensor_dump2(y);
                break;
            default:
                break;
        }
    }
}

void Constant_reshape(node_t *nd) {
    if(!nd || !nd->out) return;
}

void Constant_forward(node_t *nd) {
    if(!nd || !nd->out) return;
}

void Constant_exit(node_t *nd) {
    if(!nd || !nd->out) return;
}

void op_Constant_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Constant_init;
    nd->op->reshape     = Constant_reshape;
    nd->op->forward     = Constant_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Constant_exit;
}