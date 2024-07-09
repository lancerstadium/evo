#include "../../log.h"
#include "def.h"

static void Relu_float32(node_t *nd) {
    tensor_t *x = nd->input_tensors[0];
    tensor_t *y = nd->output_tensors[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

void op_Relu_dft(node_t *nd) {
    // Relu init
    LOG_INFO("Relu init\n");
    if (!nd || !nd->input_tensors || nd->input_tensors[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->ninput = 1) || !(nd->noutput == 1) || (nd->input_tensors[0]->ndim == 0)) {
        return;
    }
    // Relu reshape
    LOG_INFO("Relu reshape\n");

    // Relu run
    LOG_INFO("Relu run\n");
    switch (nd->input_tensors[0]->type) {
        case TENSOR_TYPE_FLOAT32:
            Relu_float32(nd);
            break;
        default:
            break;
    }

    // Relu exit
    LOG_INFO("Relu exit\n");
    return;
}