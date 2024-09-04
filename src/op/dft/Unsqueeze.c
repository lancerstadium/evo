#include <evo/resolver.h>
#include <evo/util/sys.h>
#include <string.h>


void Unsqueeze_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || (nd->in[0]->ndim == 1) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
}

void Unsqueeze_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *y = nd->out[0];
    tensor_t *x = nd->in[0];
    tensor_t *a = nd->in[1];
    int64_t *pa = (int64_t *)a->datas;
    int ndim = x->ndim + a->ndata;
    int dims[ndim];
    int axis;
    int i, j;

    memset(dims, 0, sizeof(int) * ndim);
    for (i = 0; i < a->ndata; i++) {
        axis = pa[i];
        if (axis < 0)
            axis += ndim;
        if (axis >= 0 && axis < ndim)
            dims[axis] = 1;
    }
    for (i = 0, j = 0; i < ndim; i++) {
        if (dims[i] != 1)
            dims[i] = x->dims[j++];
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
}

void Unsqueeze_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    char **px = (char **)x->datas;
    char **py = (char **)y->datas;

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

void Unsqueeze_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}


void op_Unsqueeze_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Unsqueeze_init;
    nd->op->reshape     = Unsqueeze_reshape;
    nd->op->forward     = Unsqueeze_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Unsqueeze_exit;
}