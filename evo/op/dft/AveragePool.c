#include "../../core/resolver.h"
#include "../../util/math.h"

typedef enum {
    AUTO_PAD_NOTSET = 0,
    AUTO_PAD_SAME_UPPER = 1,
    AUTO_PAD_SAME_LOWER = 2,
    AUTO_PAD_VALID = 3,
} auto_pad_t;

struct {
    auto_pad_t auto_pad;
    int ceil_mode;
    int count_include_pad;
    int* kernels;
    int nkernel;
    int* pads;
    int npad;
    int* strides;
    int nstride;
    int cpads[32];
} operator_pdata_t;


void op_AveragePool_dft(node_t*) {
    
}