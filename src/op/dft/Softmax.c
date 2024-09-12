#include <evo/resolver.h>
#include <evo/util/math.h>
#include <float.h>
#include <math.h>

typedef struct {
    int axis;
    union {
        struct {  // 13
            int caxis;
            int current;
            int outter;
            int inner;
        };
        struct {  // 11
            int N;
            int D;
        };
    };
} operator_pdata_t;

static void Softmax_forward_v13_bfloat16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float maxv, sum, v;
    int i, j, k, o, oo, io;

    for (i = 0; i < pdat->outter; i++) {
        oo = i * pdat->current * pdat->inner;
        for (k = 0; k < pdat->inner; k++) {
            io = oo + k;
            for (j = 0, maxv = bfloat16_to_float32(px[io]); j < pdat->current; j++) {
                o = io + j * pdat->inner;
                v = bfloat16_to_float32(px[o]);
                if (v > maxv)
                    maxv = v;
            }
            for (j = 0, sum = 0; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                v = expf(bfloat16_to_float32(px[o]) - maxv);
                py[o] = float32_to_bfloat16(v);
                sum += v;
            }
            if (sum != 0) {
                for (j = 0; j < pdat->current; j++) {
                    io = oo + j * pdat->inner + k;
                    v = bfloat16_to_float32(py[io]);
                    py[io] = float32_to_bfloat16(v / sum);
                }
            }
        }
    }
}

static void Softmax_forward_v13_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float maxv, sum, v;
    int i, j, k, o, oo, io;

    for (i = 0; i < pdat->outter; i++) {
        oo = i * pdat->current * pdat->inner;
        for (k = 0; k < pdat->inner; k++) {
            io = oo + k;
            for (j = 0, maxv = float16_to_float32(px[io]); j < pdat->current; j++) {
                o = io + j * pdat->inner;
                v = float16_to_float32(px[o]);
                if (v > maxv)
                    maxv = v;
            }
            for (j = 0, sum = 0; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                v = expf(float16_to_float32(px[o]) - maxv);
                py[o] = float32_to_float16(v);
                sum += v;
            }
            if (sum != 0) {
                for (j = 0; j < pdat->current; j++) {
                    io = oo + j * pdat->inner + k;
                    v = float16_to_float32(py[io]);
                    py[io] = float32_to_float16(v / sum);
                }
            }
        }
    }
}

static void Softmax_forward_v13_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    float maxv, sum;
    int i, j, k, o, oo, io;

    for (i = 0; i < pdat->outter; i++) {
        oo = i * pdat->current * pdat->inner;
        for (k = 0; k < pdat->inner; k++) {
            io = oo + k;
            for (j = 0, maxv = px[io]; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                if (px[o] > maxv)
                    maxv = px[o];
            }
            for (j = 0, sum = 0; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                py[o] = expf(px[o] - maxv);
                sum += py[o];
            }
            if (sum != 0) {
                for (j = 0; j < pdat->current; j++) {
                    io = oo + j * pdat->inner + k;
                    py[io] /= sum;
                }
            }
        }
    }
}

static void Softmax_forward_v13_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    double maxv, sum;
    int i, j, k, o, oo, io;

    for (i = 0; i < pdat->outter; i++) {
        oo = i * pdat->current * pdat->inner;
        for (k = 0; k < pdat->inner; k++) {
            io = oo + k;
            for (j = 0, maxv = px[io]; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                if (px[o] > maxv)
                    maxv = px[o];
            }
            for (j = 0, sum = 0; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                py[o] = exp(px[o] - maxv);
                sum += py[o];
            }
            if (sum != 0) {
                for (j = 0; j < pdat->current; j++) {
                    io = oo + j * pdat->inner + k;
                    py[io] /= sum;
                }
            }
        }
    }
}

static void Softmax_forward_v1_11_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float maxv, sum, v;
    int i, j, o;

    for (i = 0, o = 0; i < pdat->N; i++, o += pdat->D) {
        for (j = 0, maxv = FLT_MIN; j < pdat->D; j++) {
            v = float16_to_float32(px[o + j]);
            if (v > maxv)
                maxv = v;
        }
        for (j = 0, sum = 0; j < pdat->D; j++) {
            v = expf(float16_to_float32(px[o + j]) - maxv);
            py[o + j] = float32_to_float16(v);
            sum += v;
        }
        if (sum != 0) {
            for (j = 0; j < pdat->D; j++) {
                v = float16_to_float32(py[o + j]);
                py[o + j] = float32_to_float16(v / sum);
            }
        }
    }
}

static void Softmax_forward_v1_11_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    float maxv, sum;
    int i, j, o;

    for (i = 0, o = 0; i < pdat->N; i++, o += pdat->D) {
        for (j = 0, maxv = FLT_MIN; j < pdat->D; j++) {
            if (px[o + j] > maxv)
                maxv = px[o + j];
        }
        for (j = 0, sum = 0; j < pdat->D; j++) {
            py[o + j] = expf(px[o + j] - maxv);
            sum += py[o + j];
        }
        if (sum != 0) {
            for (j = 0; j < pdat->D; j++)
                py[o + j] /= sum;
        }
    }
}

static void Softmax_forward_v1_11_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    double maxv, sum;
    int i, j, o;

    for (i = 0, o = 0; i < pdat->N; i++, o += pdat->D) {
        for (j = 0, maxv = DBL_MIN; j < pdat->D; j++) {
            if (px[o + j] > maxv)
                maxv = px[o + j];
        }
        for (j = 0, sum = 0; j < pdat->D; j++) {
            py[o + j] = exp(px[o + j] - maxv);
            sum += py[o + j];
        }
        if (sum != 0) {
            for (j = 0; j < pdat->D; j++)
                py[o + j] /= sum;
        }
    }
}

static void Softmax_backward_v13_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];        // 输入
    tensor_t *y = nd->out[0];       // Softmax 正向传播的输出
    if (!nd->out[0]->grad) return;
    if (!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", x->name);
        nd->in[0]->grad = tensor_new(name_buf, y->type);
        tensor_reshape(nd->in[0]->grad, x->ndim, x->dims);
    }
    tensor_t *dy = nd->out[0]->grad; // 从上层传递过来的梯度（dL/dY）
    tensor_t *dx = nd->in[0]->grad;  // 需要计算的梯度（dL/dX）

    float *py = (float *)y->datas;   // Softmax输出
    float *pdy = (float *)dy->datas; // 反向传递的梯度
    float *pdx = (float *)dx->datas; // Softmax输入的梯度

    int i, j, k, o, oo, io;

    // 计算 Softmax 的反向传播梯度
    for (i = 0; i < pdat->outter; i++) {
        oo = i * pdat->current * pdat->inner;
        for (k = 0; k < pdat->inner; k++) {
            io = oo + k;

            // 计算 dx 的梯度
            for (j = 0; j < pdat->current; j++) {
                o = io + j * pdat->inner;
                // 初始化 dx 梯度为 0
                pdx[o] = 0;

                // 对每一个元素计算 Softmax 反向传播的梯度
                for (int m = 0; m < pdat->current; m++) {
                    int om = io + m * pdat->inner;

                    if (j == m) {
                        // 对角元素
                        pdx[o] += pdy[om] * py[o] * (1.0f - py[o]);
                    } else {
                        // 非对角元素
                        pdx[o] -= pdy[om] * py[o] * py[om];
                    }
                }
            }
        }
    }
}

void Softmax_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat && nd->opset >= 13) {
        pdat->axis = node_get_attr_int(nd, "axis", -1);
        nd->priv = pdat;
    } else if (pdat) {
        pdat->axis = node_get_attr_int(nd, "axis", 1);
        nd->priv = pdat;
    }
}

void Softmax_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int i;
    if (nd->opset >= 13) {
        pdat->caxis = pdat->axis;
        if (pdat->caxis < 0)
            pdat->caxis += x->ndim;
        if (pdat->caxis < 0 || pdat->caxis >= x->ndim)
            return;
        for (i = 0, pdat->outter = 1, pdat->inner = 1; i < x->ndim; i++) {
            if (i == pdat->caxis)
                pdat->current = x->dims[i];
            else if (i < pdat->caxis)
                pdat->outter *= x->dims[i];
            else
                pdat->inner *= x->dims[i];
        }
    } else {
        int axis = pdat->axis;
        if (axis < 0)
            axis += x->ndim;
        if (axis < 0 || axis >= x->ndim)
            return;
        for (i = 0, pdat->N = 1, pdat->D = 1; i < x->ndim; i++) {
            if (i < axis)
                pdat->N *= x->dims[i];
            else
                pdat->D *= x->dims[i];
        }
    }
    tensor_reshape_ident(y, x, x->type);
}

void Softmax_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (nd->opset >= 13) {
        switch (nd->in[0]->type) {
            case TENSOR_TYPE_BFLOAT16:
                Softmax_forward_v13_bfloat16(nd);
                break;
            case TENSOR_TYPE_FLOAT16:
                Softmax_forward_v13_float16(nd);
                break;
            case TENSOR_TYPE_FLOAT32:
                Softmax_forward_v13_float32(nd);
                break;
            case TENSOR_TYPE_FLOAT64:
                Softmax_forward_v13_float64(nd);
                break;
            default:
                break;
        }
    } else {
        switch (nd->in[0]->type) {
            case TENSOR_TYPE_FLOAT16:
                Softmax_forward_v1_11_float16(nd);
                break;
            case TENSOR_TYPE_FLOAT32:
                Softmax_forward_v1_11_float32(nd);
                break;
            case TENSOR_TYPE_FLOAT64:
                Softmax_forward_v1_11_float64(nd);
                break;
            default:
                break;
        }
    }
}

void Softmax_backward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (nd->opset >= 13) {
        switch (nd->in[0]->type) {
            case TENSOR_TYPE_BFLOAT16:
                // Softmax_backward_v13_bfloat16(nd);
                break;
            case TENSOR_TYPE_FLOAT16:
                // Softmax_backward_v13_float16(nd);
                break;
            case TENSOR_TYPE_FLOAT32:
                Softmax_backward_v13_float32(nd);
                break;
            case TENSOR_TYPE_FLOAT64:
                // Softmax_backward_v13_float64(nd);
                break;
            default:
                break;
        }
    } else {
        switch (nd->in[0]->type) {
            case TENSOR_TYPE_FLOAT16:
                // Softmax_backward_v1_11_float16(nd);
                break;
            case TENSOR_TYPE_FLOAT32:
                // Softmax_backward_v1_11_float32(nd);
                break;
            case TENSOR_TYPE_FLOAT64:
                // Softmax_backward_v1_11_float64(nd);
                break;
            default:
                break;
        }
    }
}

void Softmax_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Softmax_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Softmax_init;
    nd->op->reshape     = Softmax_reshape;
    nd->op->forward     = Softmax_forward;
    nd->op->backward    = Softmax_backward;
    nd->op->exit        = Softmax_exit;
}