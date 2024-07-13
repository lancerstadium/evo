#include "../../core/resolver.h"

typedef enum {
    AUTO_PAD_NOTSET = 0,
    AUTO_PAD_SAME_UPPER = 1,
    AUTO_PAD_SAME_LOWER = 2,
    AUTO_PAD_VALID = 3,
} auto_pad_t;

typedef enum {
    CONV_SIMPLE = 0,
    CONV_CACHED = 1,
    CONV_IM2COL = 2,
} conv_mode_t;

typedef struct {
    auto_pad_t auto_pad;
    int group;
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

void op_Conv_dft(node_t*) {
    
}