#include "../../core/resolver.h"
#include "../../util/math.h"
#include <math.h>
#include <string.h>

typedef enum {
    AUTO_PAD_NOTSET = 0,
    AUTO_PAD_SAME_UPPER = 1,
    AUTO_PAD_SAME_LOWER = 2,
    AUTO_PAD_VALID = 3,
} auto_pad_t;

typedef struct {
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

static inline int dim_next(int ndim, int* dims, int* dim_max) {
    if (ndim == 0)
        return 0;
    while (1) {
        ndim = ndim - 1;
        dims[ndim] += 1;
        if (dims[ndim] < dim_max[ndim])
            return 1;
        else {
            if (ndim == 0)
                return 0;
            dims[ndim] = 0;
        }
    }
}

static inline int dim_offset(int ndim, int* dims, int* dim_max) {
    int o, s;
    int i;

    for (i = ndim - 1, o = 0, s = 1; i >= 0; i--) {
        o += dims[i] * s;
        s *= dim_max[i];
    }
    return o;
}

static void AveragePool_float16(node_t* nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    uint16_t* px = (uint16_t*)x->datas;
    uint16_t* py = (uint16_t*)y->datas;
    float sum;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int padcnt, ispad, size;
    int i;

    for (i = 0, size = 1; i < x->ndim - 2; ++i)
        size *= pdat->kernels[i];
    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; i++)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        sum = 0;
        padcnt = 0;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            ispad = 0;
            for (i = 0; i < x->ndim; i++) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                    ispad = 1;
                    break;
                }
            }
            if (i >= x->ndim)
                sum += float16_to_float32(px[dim_offset(x->ndim, i_dim, x->dims)]);
            if (ispad)
                padcnt++;
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        if (pdat->count_include_pad)
            sum /= size;
        else
            sum /= (size - padcnt);
        py[dim_offset(x->ndim, o_dim, y->dims)] = float32_to_float16(sum);
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void AveragePool_float32(node_t* nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    float* px = (float*)x->datas;
    float* py = (float*)y->datas;
    float sum;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int padcnt, ispad, size;
    int i;

    for (i = 0, size = 1; i < x->ndim - 2; ++i)
        size *= pdat->kernels[i];
    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; i++)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        sum = 0;
        padcnt = 0;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            ispad = 0;
            for (i = 0; i < x->ndim; i++) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                    ispad = 1;
                    break;
                }
            }
            if (i >= x->ndim)
                sum += px[dim_offset(x->ndim, i_dim, x->dims)];
            if (ispad)
                padcnt++;
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        if (pdat->count_include_pad)
            sum /= size;
        else
            sum /= (size - padcnt);
        py[dim_offset(x->ndim, o_dim, y->dims)] = sum;
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void AveragePool_float64(node_t* nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    double* px = (double*)x->datas;
    double* py = (double*)y->datas;
    double sum;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int padcnt, ispad, size;
    int i;

    for (i = 0, size = 1; i < x->ndim - 2; ++i)
        size *= pdat->kernels[i];
    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; i++)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        sum = 0;
        padcnt = 0;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            ispad = 0;
            for (i = 0; i < x->ndim; i++) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                    ispad = 1;
                    break;
                }
            }
            if (i >= x->ndim)
                sum += px[dim_offset(x->ndim, i_dim, x->dims)];
            if (ispad)
                padcnt++;
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        if (pdat->count_include_pad)
            sum /= size;
        else
            sum /= (size - padcnt);
        py[dim_offset(x->ndim, o_dim, y->dims)] = sum;
    } while (dim_next(x->ndim, o_dim, y->dims));
}

void op_AveragePool_dft(node_t* nd) {
    // 1. AveragePool init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    int64_t* ints;
    int i, l;
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        switch (shash(node_get_attr_string(nd, "auto_pad", "NOTSET"))) {
            case 0xc3966fc2: /* "NOTSET" */
                pdat->auto_pad = AUTO_PAD_NOTSET;
                break;
            case 0xcbbc7856: /* "SAME_UPPER" */
                pdat->auto_pad = AUTO_PAD_SAME_UPPER;
                break;
            case 0xcb192d33: /* "SAME_LOWER" */
                pdat->auto_pad = AUTO_PAD_SAME_LOWER;
                break;
            case 0x0e382d15: /* "VALID" */
                pdat->auto_pad = AUTO_PAD_VALID;
                break;
            default:
                pdat->auto_pad = AUTO_PAD_NOTSET;
                break;
        }
        pdat->ceil_mode = node_get_attr_int(nd, "ceil_mode", 0);
        pdat->count_include_pad = node_get_attr_int(nd, "count_include_pad", 0);
        pdat->nkernel = node_get_attr_ints(nd, "kernel_shape", &ints);
        if (pdat->nkernel > 0) {
            pdat->kernels = malloc(sizeof(int) * pdat->nkernel);
            for (i = 0; i < pdat->nkernel; i++)
                pdat->kernels[i] = ints[i];
        }
        pdat->npad = pdat->nkernel * 2;
        pdat->pads = malloc(sizeof(int) * pdat->npad);
        if (pdat->pads) {
            l = node_get_attr_ints(nd, "pads", &ints);
            for (i = 0; i < l; i++)
                pdat->pads[i] = ints[i];
            for (; i < pdat->npad; i++)
                pdat->pads[i] = 0;
        }
        pdat->nstride = pdat->nkernel;
        pdat->strides = malloc(sizeof(int) * pdat->nstride);
        if (pdat->strides) {
            l = node_get_attr_ints(nd, "strides", &ints);
            for (i = 0; i < l; i++)
                pdat->strides[i] = ints[i];
            for (; i < pdat->nstride; i++)
                pdat->strides[i] = 1;
        }
        nd->priv = pdat;
    }
    // 2. AveragePool reshape
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    int ndim = x->ndim;
    int dims[ndim];
    int pad;
    switch (pdat->auto_pad) {
        case AUTO_PAD_NOTSET:
            memcpy(pdat->cpads, pdat->pads, sizeof(int) * pdat->npad);
            break;
        case AUTO_PAD_SAME_UPPER:
            for (i = 0; i < pdat->npad / 2; i++) {
                pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + pdat->kernels[i] - x->dims[i + 2];
                pdat->cpads[i] = pad / 2;
                pdat->cpads[i + pdat->nkernel] = pad - pdat->cpads[i];
            }
            break;
        case AUTO_PAD_SAME_LOWER:
            for (i = 0; i < pdat->npad / 2; i++) {
                pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + pdat->kernels[i] - x->dims[i + 2];
                pdat->cpads[i + pdat->nkernel] = pad / 2;
                pdat->cpads[i] = pad - pdat->cpads[i + pdat->nkernel];
            }
            break;
        case AUTO_PAD_VALID:
            memset(pdat->cpads, 0, sizeof(int) * pdat->npad);
            break;
        default:
            break;
    }
    dims[0] = x->dims[0];
    dims[1] = x->dims[1];
    for (i = 0; i < ndim - 2; i++) {
        switch (pdat->auto_pad) {
            case AUTO_PAD_NOTSET:
                if (pdat->ceil_mode)
                    dims[i + 2] = ceilf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - pdat->kernels[i]) / (float)pdat->strides[i] + 1);
                else
                    dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - pdat->kernels[i]) / (float)pdat->strides[i] + 1);
                break;
            case AUTO_PAD_SAME_UPPER:
            case AUTO_PAD_SAME_LOWER:
                dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
                break;
            case AUTO_PAD_VALID:
                dims[i + 2] = ceilf((x->dims[i + 2] - pdat->kernels[i] + 1) / (float)pdat->strides[i]);
                break;
            default:
                break;
        }
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
    // 3. AveragePool run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            AveragePool_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            AveragePool_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            AveragePool_float64(nd);
            break;
        default:
            break;
    }
    // 4. AveragePool exit
    if (pdat) {
        if (pdat->kernels)
            free(pdat->kernels);
        if (pdat->pads)
            free(pdat->pads);
        if (pdat->strides)
            free(pdat->strides);
        free(pdat);
    }
    return;
}