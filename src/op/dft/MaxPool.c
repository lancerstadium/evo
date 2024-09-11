#include <evo/resolver.h>
#include <evo/util/math.h>

#include <float.h>
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
    int storage_order;
    int *kernels;
    int nkernel;
    int *dilations;
    int ndilation;
    int *pads;
    int npad;
    int *strides;
    int nstride;
    int cpads[32];
} operator_pdata_t;

static inline int dim_next(int ndim, int *dims, int *dim_max) {
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

static inline int dim_offset(int ndim, int *dims, int *dim_max) {
    int o, s;
    int i;

    for (i = ndim - 1, o = 0, s = 1; i >= 0; i--) {
        o += dims[i] * s;
        s *= dim_max[i];
    }
    return o;
}

static void MaxPool_forward_int8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int8_t maxv, v;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int i;

    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; ++i)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        maxv = INT8_MIN;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            for (i = 0; i < x->ndim; ++i) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
                    break;
            }
            if (i >= x->ndim) {
                v = px[dim_offset(x->ndim, i_dim, x->dims)];
                maxv = MAX(v, maxv);
            }
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_forward_uint8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    uint8_t *py = (uint8_t *)y->datas;
    uint8_t maxv, v;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int i;

    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; ++i)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        maxv = 0;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            for (i = 0; i < x->ndim; ++i) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
                    break;
            }
            if (i >= x->ndim) {
                v = px[dim_offset(x->ndim, i_dim, x->dims)];
                maxv = MAX(v, maxv);
            }
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float maxv, v;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int i;

    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; ++i)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        maxv = -FLT_MAX;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            for (i = 0; i < x->ndim; ++i) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
                    break;
            }
            if (i >= x->ndim) {
                v = float16_to_float32(px[dim_offset(x->ndim, i_dim, x->dims)]);
                maxv = fmaxf(v, maxv);
            }
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        py[dim_offset(x->ndim, o_dim, y->dims)] = float32_to_float16(maxv);
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    float maxv, v;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int i;

    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; ++i)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        maxv = -FLT_MAX;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            for (i = 0; i < x->ndim; ++i) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
                    break;
            }
            if (i >= x->ndim) {
                v = px[dim_offset(x->ndim, i_dim, x->dims)];
                maxv = fmaxf(v, maxv);
            }
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    double maxv, v;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int i;

    memset(o_dim, 0, sizeof(o_dim));
    do {
        for (i = 2; i < x->ndim; ++i)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        maxv = -DBL_MAX;
        memset(k_dim, 0, sizeof(k_dim));
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            for (i = 0; i < x->ndim; ++i) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
                    break;
            }
            if (i >= x->ndim) {
                v = px[dim_offset(x->ndim, i_dim, x->dims)];
                maxv = fmax(v, maxv);
            }
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));
        py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
    } while (dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_backward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (!nd->out[0]->grad) return;

    tensor_t *x = nd->in[0];          // 输入张量 x
    tensor_t *y = nd->out[0];         // 输出张量 y
    char name_buf[54];

    // Reshape and initialize gradient tensors if necessary
    if (!nd->in[0]->grad) {
        sprintf(name_buf, "%s_grad", x->name);
        nd->in[0]->grad = tensor_new(name_buf, y->type);
        tensor_reshape(nd->in[0]->grad, x->ndim, x->dims);
    }

    float *px = (float *)x->datas;    // 输入数据
    tensor_t *dy = y->grad;           // 输出张量 y 的梯度
    float *pdx = (float *)x->grad->datas;  // 输入张量 x 的梯度
    float *pdy = (float *)dy->datas;  // 输出张量 y 的梯度

    float maxv, v;
    int k_dim[x->ndim - 2];
    int i_dim[x->ndim];
    int o_dim[x->ndim];
    int b_dim[x->ndim];
    int i;

    // 初始化输出维度
    memset(o_dim, 0, sizeof(o_dim));
    
    do {
        for (i = 2; i < x->ndim; ++i)
            b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
        
        maxv = -FLT_MAX;
        int max_idx = -1;  // 用于记录最大值的位置
        memset(k_dim, 0, sizeof(k_dim));
        
        do {
            i_dim[0] = o_dim[0];
            i_dim[1] = o_dim[1];
            for (i = 2; i < x->ndim; ++i)
                i_dim[i] = b_dim[i] + k_dim[i - 2];
            
            for (i = 0; i < x->ndim; ++i) {
                if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
                    break;
            }

            if (i >= x->ndim) {
                v = px[dim_offset(x->ndim, i_dim, x->dims)];
                if (v > maxv) {
                    maxv = v;
                    max_idx = dim_offset(x->ndim, i_dim, x->dims);  // 记录最大值的索引
                }
            }
        } while (dim_next(x->ndim - 2, k_dim, pdat->kernels));

        // 将输出梯度分配给最大值所在的输入位置
        if (max_idx != -1) {
            pdx[max_idx] += pdy[dim_offset(x->ndim, o_dim, y->dims)];
        }
        
    } while (dim_next(x->ndim, o_dim, y->dims));
}


void MaxPool_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    int64_t *ints;
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
        pdat->storage_order = node_get_attr_int(nd, "storage_order", 0);
        pdat->nkernel = node_get_attr_ints(nd, "kernel_shape", &ints);
        if (pdat->nkernel > 0) {
            pdat->kernels = malloc(sizeof(int) * pdat->nkernel);
            for (i = 0; i < pdat->nkernel; i++)
                pdat->kernels[i] = ints[i];
        }
        pdat->ndilation = pdat->nkernel;
        pdat->dilations = malloc(sizeof(int) * pdat->ndilation);
        if (pdat->dilations) {
            l = node_get_attr_ints(nd, "dilations", &ints);
            for (i = 0; i < l; i++)
                pdat->dilations[i] = ints[i];
            for (; i < pdat->ndilation; i++)
                pdat->dilations[i] = 1;
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
}

void MaxPool_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int ndim = x->ndim;
    int dims[ndim];
    int pad;
    int i;
    switch (pdat->auto_pad) {
        case AUTO_PAD_NOTSET:
            memcpy(pdat->cpads, pdat->pads, sizeof(int) * pdat->npad);
            break;
        case AUTO_PAD_SAME_UPPER:
            for (i = 0; i < pdat->npad / 2; i++) {
                pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
                pdat->cpads[i] = pad / 2;
                pdat->cpads[i + pdat->nkernel] = pad - pdat->cpads[i];
            }
            break;
        case AUTO_PAD_SAME_LOWER:
            for (i = 0; i < pdat->npad / 2; i++) {
                pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
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
                    dims[i + 2] = ceilf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
                else
                    dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
                break;
            case AUTO_PAD_SAME_UPPER:
            case AUTO_PAD_SAME_LOWER:
                dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
                break;
            case AUTO_PAD_VALID:
                dims[i + 2] = ceilf((x->dims[i + 2] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) + 1) / (float)pdat->strides[i]);
                break;
            default:
                break;
        }
    }
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
}

void MaxPool_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT32:
            MaxPool_forward_int8(nd);
            break;
        case TENSOR_TYPE_INT64:
            MaxPool_forward_uint8(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            MaxPool_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            MaxPool_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            MaxPool_forward_float64(nd);
            break;
        default:
            break;
    }
}

void MaxPool_backward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT32:
            // MaxPool_backward_int8(nd);
            break;
        case TENSOR_TYPE_INT64:
            // MaxPool_backward_uint8(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            // MaxPool_backward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            MaxPool_backward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            // MaxPool_backward_float64(nd);
            break;
        default:
            break;
    }
}

// ref: https://zhuanlan.zhihu.com/p/642116285

void MaxPool_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat) {
        if (pdat->kernels)
            free(pdat->kernels);
        if (pdat->dilations)
            free(pdat->dilations);
        if (pdat->pads)
            free(pdat->pads);
        if (pdat->strides)
            free(pdat->strides);
        free(pdat);
    }
    nd->priv = NULL;
    return;
}

void op_MaxPool_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = MaxPool_init;
    nd->op->reshape     = MaxPool_reshape;
    nd->op->forward     = MaxPool_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = MaxPool_exit;
}