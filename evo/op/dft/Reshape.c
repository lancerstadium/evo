#include "../../core/resolver.h"
#include "../../util/log.h"
#include <string.h>

void op_Reshape_dft(node_t* nd) {
    // Reshape init
    if(!nd || !nd->input_tensors  || nd->input_tensors[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if(!(nd->ninput = 2) || !(nd->noutput == 1) || (nd->input_tensors[0]->ndim == 0) || (nd->input_tensors[1]->ndim == 0)) {
        return;
    }
    // Reshape reshape
    tensor_t* y = nd->output_tensors[0];
    tensor_t* x = nd->input_tensors[0];
    tensor_t* s = nd->input_tensors[1];
    int16_t *sdata = s->datas;
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
                // LOG_INFO("%s - DIMS(%d, %d): %d\n", nd->name, i, j, x->dims[j]);
                if(sdata[j] > 0)
                    total_shape *= sdata[j];
                else if(sdata[j] == 0)
                    total_shape *= x->dims[j];
            }
            /// TODO: warning opration for reshape
            // LOG_INFO("dim: %d, shape: %d\n", total_dim, total_shape);
            // dims[i] = total_dim / total_shape;
        }
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
    // Reshape run
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
    // Reshape exit
    return;
}