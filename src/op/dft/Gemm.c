#include <evo/resolver.h>
#include <evo/util/math.h>

// Reference: https://zhuanlan.zhihu.com/p/642043155

typedef struct {
    float alpha;
    float beta;
    int transA;
    int transB;
    int m;
    int n;
    int k;
} operator_pdata_t;

static void Gemm_forward_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    int32_t *py = (int32_t *)y->datas;
    int32_t *pa = (int32_t *)a->datas;
    int32_t *pb = (int32_t *)b->datas;
    int32_t *pc;
    int32_t sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_forward_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    int64_t *py = (int64_t *)y->datas;
    int64_t *pa = (int64_t *)a->datas;
    int64_t *pb = (int64_t *)b->datas;
    int64_t *pc;
    int64_t sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_forward_uint32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    uint32_t *py = (uint32_t *)y->datas;
    uint32_t *pa = (uint32_t *)a->datas;
    uint32_t *pb = (uint32_t *)b->datas;
    uint32_t *pc;
    uint32_t sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_forward_uint64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    uint64_t *py = (uint64_t *)y->datas;
    uint64_t *pa = (uint64_t *)a->datas;
    uint64_t *pb = (uint64_t *)b->datas;
    uint64_t *pc;
    uint64_t sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_forward_bfloat16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa = (uint16_t *)a->datas;
    uint16_t *pb = (uint16_t *)b->datas;
    uint16_t *pc;
    float sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
                } else
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum);
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
                } else
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum);
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
                } else
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum);
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
                } else
                    py[oy] = float32_to_bfloat16(pdat->alpha * sum);
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    uint16_t *py = (uint16_t *)y->datas;
    uint16_t *pa = (uint16_t *)a->datas;
    uint16_t *pb = (uint16_t *)b->datas;
    uint16_t *pc;
    float sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
                } else
                    py[oy] = float32_to_float16(pdat->alpha * sum);
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
                } else
                    py[oy] = float32_to_float16(pdat->alpha * sum);
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
                } else
                    py[oy] = float32_to_float16(pdat->alpha * sum);
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
                } else
                    py[oy] = float32_to_float16(pdat->alpha * sum);
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    float *py = (float *)y->datas;
    float *pa = (float *)a->datas;
    float *pb = (float *)b->datas;
    float *pc;
    float sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

static void Gemm_backward_float32(node_t *nd) { 
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (!nd->out[0]->grad) return;

    tensor_t *a = nd->in[0];            // Input tensor a
    tensor_t *b = nd->in[1];            // Input tensor b (kernel)
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;  // Input bias c, if present
    tensor_t *y = nd->out[0];           // Output tensor y

    char name_buf[54];

    // Reshape and initialize gradient tensors if necessary
    if (!nd->in[0]->grad) {
        sprintf(name_buf, "%s_grad", a->name);
        nd->in[0]->grad = tensor_new(name_buf, y->type);
        tensor_reshape(nd->in[0]->grad, a->ndim, a->dims);
    }
    if (!nd->in[1]->grad) {
        sprintf(name_buf, "%s_grad", b->name);
        nd->in[1]->grad = tensor_new(name_buf, y->type);
        tensor_reshape(nd->in[1]->grad, b->ndim, b->dims);
    }
    if (c && !nd->in[2]->grad) {
        sprintf(name_buf, "%s_grad", c->name);
        nd->in[2]->grad = tensor_new(name_buf, y->type);
        tensor_reshape(nd->in[2]->grad, c->ndim, c->dims);
    }

    tensor_t *dy = nd->out[0]->grad;    // Output gradient (dL/dY)
    tensor_t *da = nd->in[0]->grad;  // Gradient w.r.t. 'a'
    tensor_t *db = nd->in[1]->grad;  // Gradient w.r.t. 'b' (kernel)
    tensor_t *dc = (c) ? nd->in[2]->grad : NULL;  // Gradient w.r.t. 'c', if present

    float *pdy = (float *)dy->datas;  // Gradient w.r.t. output
    float *pa = (float *)a->datas;    // Input a
    float *pb = (float *)b->datas;    // Input b (kernel)
    float *pda = (float *)da->datas;  // Gradient w.r.t. 'a'
    float *pdb = (float *)db->datas;  // Gradient w.r.t. 'b' (kernel)
    float *pdc = (dc) ? (float *)dc->datas : NULL;  // Gradient w.r.t. 'c'

    int i, j, k;
    int oy = 0;

    // Handle case where both A and B are transposed
    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[oy];
                for (k = 0; k < pdat->k; k++) {
                    pda[k * pdat->m + i] += dLdy * pb[j * pdat->k + k];  // Transpose A and B
                    pdb[j * pdat->k + k] += dLdy * pa[k * pdat->m + i];
                }
                if (pdc) pdc[oy] += dLdy;  // Update bias gradient
                oy++;
            }
        }
    }
    // Handle case where only A is transposed
    else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[oy];
                for (k = 0; k < pdat->k; k++) {
                    pda[k * pdat->m + i] += dLdy * pb[k * pdat->n + j];  // Transpose A
                    pdb[k * pdat->n + j] += dLdy * pa[k * pdat->m + i];
                }
                if (pdc) pdc[oy] += dLdy;
                oy++;
            }
        }
    }
    // Handle case where only B is transposed
    else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[oy];
                for (k = 0; k < pdat->k; k++) {
                    pda[i * pdat->k + k] += dLdy * pb[j * pdat->k + k];  // Transpose B
                    pdb[j * pdat->k + k] += dLdy * pa[i * pdat->k + k];
                }
                if (pdc) pdc[oy] += dLdy;
                oy++;
            }
        }
    }
    // Handle case where neither A nor B are transposed
    else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[oy];
                for (k = 0; k < pdat->k; k++) {
                    pda[i * pdat->k + k] += dLdy * pb[k * pdat->n + j];
                    pdb[k * pdat->n + j] += dLdy * pa[i * pdat->k + k];
                }
                if (pdc) pdc[oy] += dLdy;
                oy++;
            }
        }
    }
}



static void Gemm_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    double *py = (double *)y->datas;
    double *pa = (double *)a->datas;
    double *pb = (double *)b->datas;
    double *pc;
    double sum;
    int oa = 0;
    int ob = 0;
    int oy = 0;
    int i, j, k;

    if (pdat->transA && pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += 1;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa++;
        }
    } else if (pdat->transA) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += pdat->m;
                    ob += pdat->n;
                }
                oa -= pdat->m * pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa++;
        }
    } else if (pdat->transB) {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += 1;
                }
                oa -= pdat->k;
                ob -= pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob += pdat->k;
            }
            ob -= pdat->n * pdat->k;
            oa += pdat->k;
        }
    } else {
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                sum = 0;
                for (k = 0; k < pdat->k; k++) {
                    sum += pa[oa] * pb[ob];
                    oa += 1;
                    ob += pdat->n;
                }
                oa -= pdat->k;
                ob -= pdat->n * pdat->k;
                if (c) {
                    pc = tensor_broadcast_map_address(c, y, oy);
                    py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
                } else
                    py[oy] = pdat->alpha * sum;
                oy++;
                ob++;
            }
            ob -= pdat->n;
            oa += pdat->k;
        }
    }
}

void Gemm_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->alpha = node_get_attr_float(nd, "alpha", 0.01);
        pdat->beta = node_get_attr_float(nd, "beta", 1.0);
        pdat->transA = node_get_attr_int(nd, "transA", 0);
        pdat->transB = node_get_attr_int(nd, "transB", 0);
        pdat->m = 0;
        pdat->n = 0;
        pdat->k = 0;
        nd->priv = pdat;
    }
    nd->in[1]->is_param = 1;
    if(nd->nin > 2) nd->in[2]->is_param = 1;
}

void Gemm_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    int k;

    if (pdat->transA) {
        pdat->m = a->dims[1];
        pdat->k = a->dims[0];
    } else {
        pdat->m = a->dims[0];
        pdat->k = a->dims[1];
    }
    if (pdat->transB) {
        pdat->n = b->dims[0];
        k = 1;
    } else {
        pdat->n = b->dims[1];
        k = 0;
    }
    if (b->dims[k] != pdat->k)
        return;
    if (pdat->m <= 0 || pdat->n <= 0 || pdat->k <= 0)
        return;
    if ((nd->nin > 2) && !tensor_broadcast_is_valid(nd->in[2], (int[]){pdat->m, pdat->n}, 2))
        return;
    y->type = a->type;
    tensor_reshape(y, 2, (int[]){pdat->m, pdat->n});
}

void Gemm_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT32:
            Gemm_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Gemm_forward_int64(nd);
            break;
        case TENSOR_TYPE_UINT32:
            Gemm_forward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            Gemm_forward_uint64(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Gemm_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Gemm_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Gemm_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Gemm_forward_float64(nd);
            break;
        default:
            break;
    }
}

void Gemm_backward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_INT32:
            // Gemm_backward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            // Gemm_backward_int64(nd);
            break;
        case TENSOR_TYPE_UINT32:
            // Gemm_backward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            // Gemm_backward_uint64(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            // Gemm_backward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            // Gemm_backward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Gemm_backward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            // Gemm_backward_float64(nd);
            break;
        default:
            break;
    }
}

void Gemm_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Gemm_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Gemm_init;
    nd->op->reshape     = Gemm_reshape;
    nd->op->forward     = Gemm_forward;
    nd->op->backward    = Gemm_backward;
    nd->op->exit        = Gemm_exit;
}