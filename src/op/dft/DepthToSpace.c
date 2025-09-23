#include <evo/resolver.h>
#include <string.h>

typedef struct {
    int64_t blocksize;
    char* mode;
} operator_pdata_t;

void DepthToSpace_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if(pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->blocksize = node_get_attr_int(nd, "blocksize", 1);
        pdat->mode = node_get_attr_string(nd, "mode", "DCR");
        nd->priv = pdat;
    }
}

void DepthToSpace_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->ndim != 4) {
        return;
    }
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    operator_pdata_t* pdat = nd->priv;
    y->type = x->type;
    if(nd->in[0]->layout == 0) {
        tensor_reshape(y, 4, (int[]){x->dims[0], x->dims[1] / (int)(pdat->blocksize * pdat->blocksize), x->dims[2] * (int)pdat->blocksize, x->dims[3] * (int)pdat->blocksize});
    } else if(nd->in[0]->layout == 1) {
        tensor_reshape(y, 4, (int[]){x->dims[0], x->dims[1] * (int)pdat->blocksize, x->dims[2] * (int)pdat->blocksize, x->dims[3] / (int)(pdat->blocksize * pdat->blocksize)});
    }
}

void DepthToSpace_forward(node_t *nd) {
    if (!nd || !nd->in || !nd->out) return;

    // Input validation: ensure the input tensor is 4D and properly defined
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0)
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED
        || nd->in[0]->ndim != 4) {
        return;
    }

    tensor_t *y = nd->out[0];  // Output tensor
    tensor_t *x = nd->in[0];   // Input tensor
    void *px = x->datas;       // Input data pointer
    void *py = y->datas;       // Output data pointer
    int sz = tensor_type_sizeof(x->type);  // Data element size in bytes
    operator_pdata_t *pdat = nd->priv;

    // Block size check: if block size is 1, just copy the input tensor to the output
    if (pdat->blocksize == 1 && px && py) {
        memcpy(py, px, x->ndata * sz);
        return;
    }

    int bsz = pdat->blocksize;  // Block size

    // Only NCHW layout (channel-first layout) considered here
    if (nd->in[0]->layout == 0) {
        int N = x->dims[0];      // Batch size
        int C = x->dims[1];      // Input channels (C should be divisible by blocksize^2)
        int H = x->dims[2];      // Input height
        int W = x->dims[3];      // Input width
        int new_C = C / (bsz * bsz);  // New number of channels after depth rearrangement
        int new_H = H * bsz;     // New height after space rearrangement
        int new_W = W * bsz;     // New width after space rearrangement

        // Iterate over batch
        for (int n = 0; n < N; ++n) {
            // Iterate over new channels
            for (int c = 0; c < new_C; ++c) {
                // Iterate over the original height and width
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        // For each spatial location, distribute values into the new spatial locations
                        for (int i = 0; i < bsz; ++i) {  // block height (row direction)
                            for (int j = 0; j < bsz; ++j) {  // block width (column direction)
                                int input_idx = n * (C * H * W) + (c * bsz * bsz + i * bsz + j) * (H * W) + h * W + w;
                                int output_idx = n * (new_C * new_H * new_W) + c * (new_H * new_W) + (h * bsz + i) * new_W + (w * bsz + j);
                                memcpy((char *)py + output_idx * sz, (char *)px + input_idx * sz, sz);
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}


void DepthToSpace_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_DepthToSpace_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = DepthToSpace_init;
    nd->op->reshape     = DepthToSpace_reshape;
    nd->op->forward     = DepthToSpace_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = DepthToSpace_exit;
}