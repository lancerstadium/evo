#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>

typedef struct {
    int axis;
    int keepdims;
    int select_last_index;

    int dim;
    int stride;
} operator_pdata_t;

static void ArgMax_forward_int8(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int8_t *p, *px = x->datas;
    int8_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_int16(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int16_t *p, *px = x->datas;
    int16_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_int32(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int32_t *p, *px = x->datas;
    int32_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_int64(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int64_t *p, *px = x->datas;
    int64_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_uint8(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *p, *px = x->datas;
    uint8_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_uint16(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *p, *px = x->datas;
    uint16_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_uint32(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint32_t *p, *px = x->datas;
    uint32_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_uint64(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint64_t *p, *px = x->datas;
    uint64_t maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_bfloat16(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *p, *px = x->datas;
    float maxv, v;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = bfloat16_to_float32(px[idx]), maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                v = bfloat16_to_float32(*p);
                if (pdat->select_last_index) {
                    if (v >= maxv) {
                        maxv = v;
                        maxi = i;
                    }
                } else {
                    if (v > maxv) {
                        maxv = v;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_float16(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *p, *px = x->datas;
    float maxv, v;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = float16_to_float32(px[idx]), maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                v = float16_to_float32(*p);
                if (pdat->select_last_index) {
                    if (v >= maxv) {
                        maxv = v;
                        maxi = i;
                    }
                } else {
                    if (v > maxv) {
                        maxv = v;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_float32(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *p, *px = x->datas;
    float maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

static void ArgMax_forward_float64(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *p, *px = x->datas;
    double maxv;
    int64_t *py = y->datas;
    int64_t maxi;
    size_t len = x->ndata;
    size_t idx = 0;
    int cnt = 0;
    int i;

    while (idx < len) {
        if (cnt < pdat->stride) {
            for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
                if (pdat->select_last_index) {
                    if (*p >= maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                } else {
                    if (*p > maxv) {
                        maxv = *p;
                        maxi = i;
                    }
                }
            }
            *py++ = maxi;
            idx++;
            cnt++;
        } else {
            idx += (pdat->dim - 1) * pdat->stride;
            cnt = 0;
        }
    }
}

void ArgMax_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->axis = node_get_attr_int(nd, "axis", 0);
        pdat->keepdims = node_get_attr_int(nd, "keepdims", 1);
        pdat->select_last_index = node_get_attr_int(nd, "select_last_index", 0);
        nd->priv = pdat;
    }
    return;
}

void ArgMax_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int axis = pdat->axis;
    int ndim = x->ndim;
    int dims[ndim];
    int i;

    if (axis < 0)
        axis += x->ndim;
    if (axis < 0 || axis >= x->ndim)
        return;
    pdat->dim = x->dims[axis];
    pdat->stride = x->strides[axis];
    if (pdat->keepdims) {
        memcpy(dims, x->dims, sizeof(int) * ndim);
        dims[axis] = 1;
    } else {
        for (i = 0, ndim = 0; i < x->ndim; i++) {
            if (i != axis)
                dims[ndim++] = x->dims[i];
        }
    }
    y->type = TENSOR_TYPE_INT64;
    tensor_reshape(y, ndim, dims);
}

void ArgMax_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT8:
            ArgMax_forward_int8(nd);
            break;
        case TENSOR_TYPE_INT16:
            ArgMax_forward_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            ArgMax_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            ArgMax_forward_int64(nd);
            break;
        case TENSOR_TYPE_UINT8:
            ArgMax_forward_uint8(nd);
            break;
        case TENSOR_TYPE_UINT16:
            ArgMax_forward_uint16(nd);
            break;
        case TENSOR_TYPE_UINT32:
            ArgMax_forward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            ArgMax_forward_uint64(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            ArgMax_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            ArgMax_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            ArgMax_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            ArgMax_forward_float64(nd);
            break;
        default:
            break;
    }
}

void ArgMax_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_ArgMax_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = ArgMax_init;
    nd->op->reshape     = ArgMax_reshape;
    nd->op->forward     = ArgMax_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = ArgMax_exit;
}