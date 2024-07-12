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



void op_MaxPool_dft(node_t*) {
    // 1. MaxPool init

    // 2. MaxPool reshape

    // 3. MaxPool run

    // 4. MaxPool exit
}