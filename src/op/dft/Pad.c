#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>


typedef struct {
    char* mode;
} operator_pdata_t;

void Pad_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->mode = node_get_attr_string(nd, "mode", "constant");
        nd->priv = pdat;
    }
}

#include <evo/util/log.h>

// ref: https://onnx.ai/onnx/operators/onnx__Pad.html
void Pad_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(nd->nin < 2 || nd->in[1]->type != TENSOR_TYPE_INT64) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t* a = nd->in[0];    /* data */
    tensor_t* b = nd->in[1];    /* pads */
    tensor_t* y = nd->out[0];   /* outs */
    int64_t* pads = b->datas;
    switch(shash(pdat->mode)) {
        case 0x42a2b30f:        /* constant */
        case 0x3e3a6a0a:        /* reflect  */
        case 0x7c9628da: {      /* edge     */
            int ndim = a->ndim;
            int dims[ndim];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            for(int i = 0; i < a->ndim; i++) {
                dims[i] = a->dims[i] + pads[i] + pads[i + ndim];
            }
            y->type = a->type;
            tensor_reshape(y, ndim, dims);
        }
        default: break;
    }
}

void Pad_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if(nd->nin < 2 || nd->in[1]->type != TENSOR_TYPE_INT64) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t* a = nd->in[0];    /* data */
    tensor_t* b = nd->in[1];    /* pads */
    tensor_t* y = nd->out[0];   /* outs */
    int64_t* pads = b->datas;
    switch(shash(pdat->mode)) {
        case 0x42a2b30f: {      /* constant */
            void* constant_value = NULL;
            /* Get constant value for padding if provided */
            if (nd->nin >= 3 && nd->in[2]->type == a->type) {
                constant_value = nd->in[2]->datas;
            } else {
                /* If no constant value provided, default to 0 */
                constant_value = calloc(1, tensor_type_sizeof(a->type));
            }
            /* Initialize all output to the constant value */
            for (int i = 0; i < y->ndata; i++) {
                char *output_ptr = (char *)y->datas + i * tensor_type_sizeof(a->type);
                memcpy(output_ptr, constant_value, tensor_type_sizeof(a->type));
            }
            /* Iterate over each element of the input tensor and copy to the output tensor */
            int ndim = a->ndim;
            int* input_pos = calloc(ndim, sizeof(int));    /* To track input position in a multi-dimensional tensor */
            int* output_pos = calloc(ndim, sizeof(int));   /* To track output position after applying padding */

            for (int i = 0; i < a->ndata; i++) {
                /* Convert flat index to multi-dimensional indices for input */
                int flat_idx = i;
                for (int d = ndim - 1; d >= 0; d--) {
                    input_pos[d] = flat_idx % a->dims[d];
                    flat_idx /= a->dims[d];
                    /* Compute output position by adding corresponding padding for each dimension */
                    output_pos[d] = input_pos[d] + pads[d];
                }

                /* Compute the linearized memory addresses for input and output */
                char* input_ptr = (char *)a->datas;
                char* output_ptr = (char *)y->datas;

                for (int d = 0; d < ndim; d++) {
                    input_ptr += input_pos[d] * a->strides[d] * tensor_type_sizeof(a->type);
                    output_ptr += output_pos[d] * y->strides[d] * tensor_type_sizeof(a->type);
                }

                /* Copy input data to the correct position in the output tensor */
                memcpy(output_ptr, input_ptr, tensor_type_sizeof(a->type));
            }

            /* Free allocated memory for tracking positions */
            free(input_pos);
            free(output_pos);
            /* Free default constant value if allocated */
            if (nd->nin < 3 || nd->in[2]->type != a->type) {
                free(constant_value);
            }
            break;
        }
        default: break;
    }
}

void Pad_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Pad_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Pad_init;
    nd->op->reshape     = Pad_reshape;
    nd->op->forward     = Pad_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Pad_exit;
}