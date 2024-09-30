#include <evo/resolver.h>
#include <evo/util/math.h>

// Reference: https://zhuanlan.zhihu.com/p/642043155

typedef enum {
    BROADCAST_NONE,
    BROADCAST_SCALAR,
    BROADCAST_ROW_VECTOR,
    BROADCAST_COL_VECTOR,
    BROADCAST_MATRIX
} broadcast_type_t;

typedef struct {
    float alpha;
    float beta;
    int transA;
    int transB;
    int m;
    int n;
    int k;
    broadcast_type_t bc;
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

#include <evo/dev/cuda/def.h>

static void Gemm_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;
    float *py = (float *)y->datas;
    float *pa = (float *)a->datas;
    float *pb = (float *)b->datas;
    float *pc = (c != NULL) ? (float *)c->datas : NULL;  // 偏置矩阵 C 的数据指针（如果存在）

    // float sum;
    // int oa = 0;
    // int ob = 0;
    // int oy = 0;
    // int i, j, k;

    Gemm_forward_float32_cuda(pa, pb, pc, py, pdat->alpha, pdat->beta, pdat->m, pdat->n, pdat->k, pdat->transA, pdat->transB, pdat->bc);

    // if (pdat->transA && pdat->transB) {
    //     for (i = 0; i < pdat->m; i++) {
    //         for (j = 0; j < pdat->n; j++) {
    //             sum = 0;
    //             for (k = 0; k < pdat->k; k++) {
    //                 sum += pa[oa] * pb[ob];
    //                 oa += pdat->m;
    //                 ob += 1;
    //             }
    //             oa -= pdat->m * pdat->k;
    //             ob -= pdat->k;
    //             if (c) {
    //                 pc = tensor_broadcast_map_address(c, y, oy);
    //                 py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
    //             } else
    //                 py[oy] = pdat->alpha * sum;
    //             oy++;
    //             ob += pdat->k;
    //         }
    //         ob -= pdat->n * pdat->k;
    //         oa++;
    //     }
    // } else if (pdat->transA) {
    //     for (i = 0; i < pdat->m; i++) {
    //         for (j = 0; j < pdat->n; j++) {
    //             sum = 0;
    //             for (k = 0; k < pdat->k; k++) {
    //                 sum += pa[oa] * pb[ob];
    //                 oa += pdat->m;
    //                 ob += pdat->n;
    //             }
    //             oa -= pdat->m * pdat->k;
    //             ob -= pdat->n * pdat->k;
    //             if (c) {
    //                 pc = tensor_broadcast_map_address(c, y, oy);
    //                 py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
    //             } else
    //                 py[oy] = pdat->alpha * sum;
    //             oy++;
    //             ob++;
    //         }
    //         ob -= pdat->n;
    //         oa++;
    //     }
    // } else if (pdat->transB) {
    //     for (i = 0; i < pdat->m; i++) {
    //         for (j = 0; j < pdat->n; j++) {
    //             sum = 0;
    //             for (k = 0; k < pdat->k; k++) {
    //                 sum += pa[oa] * pb[ob];
    //                 oa += 1;
    //                 ob += 1;
    //             }
    //             oa -= pdat->k;
    //             ob -= pdat->k;
    //             if (c) {
    //                 pc = tensor_broadcast_map_address(c, y, oy);
    //                 py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
    //             } else
    //                 py[oy] = pdat->alpha * sum;
    //             oy++;
    //             ob += pdat->k;
    //         }
    //         ob -= pdat->n * pdat->k;
    //         oa += pdat->k;
    //     }
    // } else {
    //     for (i = 0; i < pdat->m; i++) {
    //         for (j = 0; j < pdat->n; j++) {
    //             sum = 0;
    //             for (k = 0; k < pdat->k; k++) {
    //                 sum += pa[oa] * pb[ob];
    //                 oa += 1;
    //                 ob += pdat->n;
    //             }
    //             oa -= pdat->k;
    //             ob -= pdat->n * pdat->k;
    //             if (c) {
    //                 pc = tensor_broadcast_map_address(c, y, oy);
    //                 py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
    //             } else
    //                 py[oy] = pdat->alpha * sum;
    //             oy++;
    //             ob++;
    //         }
    //         ob -= pdat->n;
    //         oa += pdat->k;
    //     }
    // }
}

static void Gemm_backward_float32(node_t *nd) { 
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *a = nd->in[0];  // 输入tensor a
    tensor_t *b = nd->in[1];  // 输入tensor b（kernel）
    tensor_t *c = (nd->nin > 2) ? nd->in[2] : NULL;  // 输入bias c, 如果存在
    // tensor_t *y = nd->out[0];  // 输出tensor y

    tensor_t *dy = nd->out[0]->grad;  // 输出梯度 (dL/dY)
    tensor_t *da = nd->in[0]->grad;  // 梯度 w.r.t. 'a'
    tensor_t *db = nd->in[1]->grad;  // 梯度 w.r.t. 'b' (kernel)
    tensor_t *dc = (c) ? nd->in[2]->grad : NULL;  // 梯度 w.r.t. 'c', 如果存在

    float *pdy = (float *)dy->datas;  // 梯度 w.r.t. 输出
    float *pa = (float *)a->datas;  // 输入 a
    float *pb = (float *)b->datas;  // 输入 b (kernel)
    float *pda = (float *)da->datas;  // 梯度 w.r.t. 'a'
    float *pdb = (float *)db->datas;  // 梯度 w.r.t. 'b' (kernel)
    float *pdc = (dc) ? (float *)dc->datas : NULL;  // 梯度 w.r.t. 'c'

    int i, j, k;

    if (pdat->transA && pdat->transB) {
        // A 和 B 都转置的情况
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[i * pdat->n + j];
                for (k = 0; k < pdat->k; k++) {
                    pda[k * pdat->m + i] += dLdy * pb[j * pdat->k + k];  // 更新 dA
                    pdb[j * pdat->k + k] += dLdy * pa[k * pdat->m + i];  // 更新 dB
                }
                if (pdc) pdc[j] += dLdy;  // 更新 bias 梯度
            }
        }
    }
    else if (pdat->transA) {
        // 只有 A 转置的情况
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[i * pdat->n + j];
                for (k = 0; k < pdat->k; k++) {
                    pda[k * pdat->m + i] += dLdy * pb[k * pdat->n + j];  // 更新 dA
                    pdb[k * pdat->n + j] += dLdy * pa[k * pdat->m + i];  // 更新 dB
                }
                if (pdc) pdc[j] += dLdy;
            }
        }
    }
    else if (pdat->transB) {
        // 只有 B 转置的情况
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[i * pdat->n + j];
                for (k = 0; k < pdat->k; k++) {
                    pda[i * pdat->k + k] += dLdy * pb[j * pdat->k + k];  // 更新 dA
                    pdb[j * pdat->k + k] += dLdy * pa[i * pdat->k + k];  // 更新 dB
                }
                if (pdc) pdc[j] += dLdy;
            }
        }
    }
    else {
        // A 和 B 都不转置的情况
        for (i = 0; i < pdat->m; i++) {
            for (j = 0; j < pdat->n; j++) {
                float dLdy = pdy[i * pdat->n + j];
                for (k = 0; k < pdat->k; k++) {
                    pda[i * pdat->k + k] += dLdy * pb[k * pdat->n + j];  // 更新 dA
                    pdb[k * pdat->n + j] += dLdy * pa[i * pdat->k + k];  // 更新 dB
                }
                if (pdc) pdc[j] += dLdy;
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
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->alpha = node_get_attr_float(nd, "alpha", 1.0);
        pdat->beta = node_get_attr_float(nd, "beta", 1.0);
        pdat->transA = node_get_attr_int(nd, "transA", 0);
        pdat->transB = node_get_attr_int(nd, "transB", 0);
        pdat->m = 0;
        pdat->n = 0;
        pdat->k = 0;
        nd->priv = pdat;
    }
}

void Gemm_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    nd->in[1]->is_param = 1;
    if(nd->nin > 2) nd->in[2]->is_param = 1;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *a = nd->in[0];
    tensor_t *b = nd->in[1];
    pdat->m = pdat->transA ? nd->in[0]->dims[1] : nd->in[0]->dims[0];
    pdat->n = pdat->transB ? nd->in[1]->dims[0] : nd->in[1]->dims[1];
    pdat->k = pdat->transA ? nd->in[0]->dims[0] : nd->in[0]->dims[1];
    int k = pdat->transB ? 1 : 0;
    if (b->dims[k] != pdat->k)
        return;
    if (pdat->m <= 0 || pdat->n <= 0 || pdat->k <= 0)
        return;
    // if ((nd->nin > 2) && !tensor_broadcast_is_valid(nd->in[2], (int[]){pdat->m, pdat->n}, 2))
    //     return;

    // 检查 C 的广播规则
    if (nd->nin > 2) {
        tensor_t *c = nd->in[2];
        if (!tensor_broadcast_is_valid(c, (int[]){pdat->m, pdat->n}, 2))
            return;

        // 在预处理阶段确定 C 的广播类型
        if (c->ndim == 0) {
            pdat->bc = BROADCAST_SCALAR;  // C 是标量
        } else if (c->dims[0] == 1 && c->dims[1] == pdat->n) {
            pdat->bc = BROADCAST_ROW_VECTOR;  // C 是行向量
        } else if (c->dims[0] == pdat->m && c->dims[1] == 1) {
            pdat->bc = BROADCAST_COL_VECTOR;  // C 是列向量
        } else {
            pdat->bc = BROADCAST_MATRIX;  // C 是完整矩阵
        }
    } else {
        pdat->bc = BROADCAST_NONE;  // 没有 C
    }


    y->type = a->type;
    tensor_reshape(y, 2, (int[]){pdat->m, pdat->n});
}

void Gemm_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
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
    if(!nd->out[0]->grad) return;
    if(!nd->in[0]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->in[0]->name);
        nd->in[0]->grad = tensor_new(name_buf, nd->in[0]->type);
        tensor_reshape(nd->in[0]->grad, nd->in[0]->ndim, nd->in[0]->dims);
    }
    if (!nd->in[1]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->in[1]->name);
        nd->in[1]->grad = tensor_new(name_buf, nd->in[1]->type);
        tensor_reshape(nd->in[1]->grad, nd->in[1]->ndim, nd->in[1]->dims);
    }
    if (nd->nin > 2 && !nd->in[2]->grad) {
        char name_buf[54];
        sprintf(name_buf, "%s_grad", nd->in[2]->name);
        nd->in[2]->grad = tensor_new(name_buf, nd->in[2]->type);
        tensor_reshape(nd->in[2]->grad, nd->in[2]->ndim, nd->in[2]->dims);
    }
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