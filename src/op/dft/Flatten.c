#include <evo/resolver.h>
#include <evo/util/sys.h>
#include <string.h>

typedef struct {
    int axis;
} operator_pdata_t;


void Flatten_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->axis = node_get_attr_int(nd, "axis", 1);
        nd->priv = pdat;
    }
}

void Flatten_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
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
}

void Flatten_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(nd->in[0]->type == TENSOR_TYPE_UNDEFINED) return;
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

void Flatten_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}



void op_Flatten_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Flatten_init;
    nd->op->reshape     = Flatten_reshape;
    nd->op->forward     = Flatten_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Flatten_exit;
}