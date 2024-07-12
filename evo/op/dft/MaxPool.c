#include <math.h>
#include "../../core/resolver.h"

typedef enum {
    AUTO_PAD_NOTSET = 0,
    AUTO_PAD_SAME_UPPER = 1,
    AUTO_PAD_SAME_LOWER = 2,
    AUTO_PAD_VALID = 3,
} auto_pad_t;

typedef struct {
    auto_pad_t auto_pad;
    int ceil_mode;
    int storage_order;
    int* kernels;
    int nkernel;
    int* dilations;
    int ndilation;
    int* pads;
    int npad;
    int* strides;
    int nstride;
    int cpads[32];
} operator_pdata_t;



void op_MaxPool_dft(node_t *nd) {
    // 1. MaxPool init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    // operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    // int64_t* ints;
    // int i, l;
    // 2. MaxPool reshape

    // 3. MaxPool run

    // 4. MaxPool exit
}