#include <evo/resolver.h>
#include <evo/util/log.h>
#include <string.h>


void Reshape_init(node_t* nd) {
    if(!nd || !nd->in) {
        return;
    }
    if(!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
}

void Reshape_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    tensor_t* s = nd->in[1];
    int64_t *sdata = s->datas;
	int total_dim = 1;
	int total_shape = 1;
    int ndim = s->ndata;
    int dims[ndim];
    for(int i = 0; i < ndim; i++) {
        if(sdata[i] == 0)
            dims[i] = x->dims[i];
        else if(sdata[i] > 0)
            dims[i] = sdata[i];
        else {
            for(int j = 0; j < x->ndim; j++)
                total_dim *= x->dims[j];
            for(int j = 0; j < ndim; j++) {
                if(sdata[j] > 0)
                    total_shape *= sdata[j];
                else if(sdata[j] == 0)
                    total_shape *= x->dims[j];
            }
            dims[i] = total_dim / total_shape;
        }
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
}

void Reshape_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    char** xdata = x->datas;
    char** ydata = y->datas;
    if(x->type == TENSOR_TYPE_STRING) {
        for(int i = 0; i < y->ndata; i++) {
            if(ydata[i])
                free(ydata[i]);
            ydata[i] = xdata[i];
        }
    } else {
        memcpy(ydata, xdata, x->ndata * tensor_type_sizeof(x->type));
    }
}

void Reshape_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Reshape_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Reshape_init;
    nd->op->reshape     = Reshape_reshape;
    nd->op->forward     = Reshape_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Reshape_exit;
}