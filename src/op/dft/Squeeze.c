#include <evo/resolver.h>
#include <string.h>


void Squeeze_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void Squeeze_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
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
}

void Squeeze_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
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

void Squeeze_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Squeeze_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Squeeze_init;
    nd->op->reshape     = Squeeze_reshape;
    nd->op->forward     = Squeeze_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Squeeze_exit;
}