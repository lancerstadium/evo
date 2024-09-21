#include <evo/resolver.h>
#include <evo/util/math.h>

#include <string.h>
#include <math.h>

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

static inline void dgemm_float32(int n, int m, int o, float* A, float* B, float* C) {
    typedef float(*atype)[o];
    typedef float(*btype)[m];
    typedef float(*ctype)[m];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            ((ctype)C)[i][j] = 0.;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < o; ++k) {
            for (int j = 0; j < m; ++j) {
                ((ctype)C)[i][j] += ((atype)A)[i][k] * ((btype)B)[k][j];
            }
        }
    }
}

static inline void dgemm_float64(int n, int m, int o, double* A, double* B, double* C) {
    typedef double(*atype)[o];
    typedef double(*btype)[m];
    typedef double(*ctype)[m];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            ((ctype)C)[i][j] = 0.;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < o; ++k) {
            for (int j = 0; j < m; ++j) {
                ((ctype)C)[i][j] += ((atype)A)[i][k] * ((btype)B)[k][j];
            }
        }
    }
}

static void Conv_forward_float16(node_t *nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    tensor_t* w = nd->in[1];
    tensor_t* b = NULL;
    uint16_t* py = (uint16_t*)y->datas;
    uint16_t* px = (uint16_t*)x->datas;
    uint16_t* pw = (uint16_t*)w->datas;
    uint16_t* pb = NULL;

    conv_mode_t conv_mode = CONV_SIMPLE;
    float* pxcache = NULL;
    float* matw = NULL;
    float* matx = NULL;
    float* maty = NULL;

    float sum, v, weight;
    int ndim = x->ndim;
    int M = w->dims[0];
    int C = w->dims[1];
    int H = w->dims[2];
    int W = w->dims[3];
    int ch, i;

    if (nd->nin > 2) {
        b = nd->in[2];
        pb = (uint16_t*)b->datas;
    }
    if (ndim == 4) {
        int iC = x->dims[1];
        int iH = x->dims[2];
        int iW = x->dims[3];

        int oN = y->dims[0];
        int oC = w->dims[0];
        int oH = y->dims[2];
        int oW = y->dims[3];

        int MM = M / pdat->group;
        int CC = iC / pdat->group;

        typedef float(*pxtype)[iC][iH][iW];
        typedef float(*pwtype)[C][H][W];
        typedef float(*pytype)[M][oH][oW];
        typedef float(*pxcachetype)[(oC * pdat->group / M) * C][H][W];
        typedef float(*mwtype) /*[H * W * C]*/[MM];
        typedef float(*mxtype) /*[oH * oW]*/[H * W * C];
        typedef float(*mytype) /*[oH * oW]*/[MM];

        /* try im2col first */
        matw = malloc(MM * H * W * C * sizeof(float));
        matx = malloc(oH * oW * H * W * C * sizeof(float));
        maty = malloc(oH * oW * MM * sizeof(float));
        if (matw && matx && maty) {
            conv_mode = CONV_IM2COL;
        } else {
            if (matw) free(matw);
            if (matx) free(matx);
            if (maty) free(maty);

            /* then try cached conv */
            pxcache = malloc(oN * (oC * pdat->group / M) * C * H * W * sizeof(float));
            if (pxcache) {
                conv_mode = CONV_CACHED;
            }
        }

        if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED) {
            for (int h = 0; h < oH; ++h) {
                for (int w = 0; w < oW; ++w) {
                    int base_h = h * pdat->strides[0] - pdat->cpads[0];
                    int base_w = w * pdat->strides[1] - pdat->cpads[1];

                    if (pxcache) {
                        for (int n = 0; n < oN; ++n) {
                            for (int group_c = 0; group_c < oC * pdat->group / M; ++group_c) {
                                int base_c = group_c * C;
                                for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
                                    int input_h = base_h + i * pdat->dilations[0];
                                    if (input_h >= iH)
                                        break;
                                    for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
                                        int input_w = base_w + j * pdat->dilations[1];
                                        if (input_w >= iW)
                                            break;
                                        for (int w_channel = 0; w_channel < C; ++w_channel) {
                                            ch = base_c + w_channel;
                                            ((pxcachetype)pxcache)[n][ch][i][j] = float16_to_float32(((pxtype)px)[n][ch][input_h][input_w]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int n = 0; n < oN; ++n) {
                        for (int c = 0; c < oC; ++c) {
                            int base_c = (c * pdat->group / M) * C;
                            sum = 0;
                            for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
                                int input_h = base_h + i * pdat->dilations[0];
                                if (input_h >= iH)
                                    break;
                                for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
                                    int input_w = base_w + j * pdat->dilations[1];
                                    if (input_w >= iW)
                                        break;
                                    for (int w_channel = 0; w_channel < C; ++w_channel) {
                                        ch = base_c + w_channel;
                                        if (pxcache) {
                                            v = ((pxcachetype)pxcache)[n][ch][i][j];
                                        } else {
                                            v = float16_to_float32(((pxtype)px)[n][ch][input_h][input_w]);
                                        }
                                        weight = float16_to_float32(((pwtype)pw)[c][w_channel][i][j]);
                                        sum += v * weight;
                                    }
                                }
                            }
                            if (pb)
                                sum += float16_to_float32(pb[c]);
                            ((pytype)py)[n][c][h][w] = float32_to_float16(sum);
                        }
                    }
                }
            }
            if (pxcache) {
                free(pxcache);
            }
        } else if (conv_mode == CONV_IM2COL) {
            for (int g = 0; g < pdat->group; g++) {
                for (size_t m = 0; m < MM; m++) {
                    for (size_t c = 0; c < C; c++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                ((mwtype)matw)[c * H * W + h * W + w][m] = float16_to_float32(((pwtype)pw)[g * MM + m][c][h][w]);
                            }
                        }
                    }
                }

                for (int n = 0; n < oN; n++) {
                    for (size_t hh = 0; hh < oH; hh++) {
                        for (size_t ww = 0; ww < oW; ww++) {
                            int base_h = hh * pdat->strides[0] - pdat->cpads[0];
                            int base_w = ww * pdat->strides[1] - pdat->cpads[1];
                            for (size_t c = 0; c < C; c++) {
                                for (size_t h = 0; h < H; h++) {
                                    for (size_t w = 0; w < W; w++) {
                                        int ih = base_h + h * pdat->dilations[0];
                                        int iw = base_w + w * pdat->dilations[1];
                                        if (ih < 0 || iw < 0 || ih >= iH || iw >= iW) {
                                            ((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = 0.;
                                        } else {
                                            ((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = float16_to_float32(((pxtype)px)[n][g * CC + c][ih][iw]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    dgemm_float32(oH * oW, MM, H * W * C, matx, matw, maty);
                    for (int m = 0; m < MM; ++m) {
                        for (int h = 0; h < oH; ++h) {
                            for (int w = 0; w < oW; ++w) {
                                float t = ((mytype)maty)[h * oW + w][m];
                                if (pb) {
                                    t += float16_to_float32(pb[g * MM + m]);
                                }
                                ((pytype)py)[n][g * MM + m][h][w] = float32_to_float16(t);
                            }
                        }
                    }
                }
            }
            free(matw);
            free(matx);
            free(maty);
        } else {
            /* never */
        }
    } else {
        int i_dim[ndim];
        int o_dim[ndim];
        int w_dim[ndim];
        int b_dim[ndim];

        memset(o_dim, 0, sizeof(o_dim));
        do {
            b_dim[0] = o_dim[0];
            for (i = 2; i < ndim; i++)
                b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
            sum = 0;
            memset(w_dim, 0, sizeof(w_dim));
            w_dim[0] = o_dim[1];
            do {
                if (w_dim[1] == 1)
                    break;
                i_dim[0] = b_dim[0];
                for (i = 2; i < ndim; i++)
                    i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
                for (ch = 0; ch < C; ch++) {
                    i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
                    w_dim[1] = ch;
                    for (i = 0; i < ndim; i++) {
                        if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                            v = 0;
                            break;
                        }
                    }
                    if (i >= ndim)
                        v = float16_to_float32(px[dim_offset(ndim, i_dim, x->dims)]);
                    for (i = 0; i < ndim; i++) {
                        if ((w_dim[i] < 0) || (w_dim[i] >= w->dims[i])) {
                            weight = 0;
                            break;
                        }
                    }
                    if (i >= ndim)
                        weight = float16_to_float32(pw[dim_offset(ndim, w_dim, w->dims)]);
                    sum += v * weight;
                }
                w_dim[1] = 0;
            } while (dim_next(ndim, w_dim, w->dims));
            if (pb)
                sum += float16_to_float32(pb[o_dim[1]]);
            py[dim_offset(ndim, o_dim, y->dims)] = float32_to_float16(sum);
        } while (dim_next(ndim, o_dim, y->dims));
    }
}

static void Conv_forward_float32(node_t *nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    tensor_t* w = nd->in[1];
    tensor_t* b = NULL;
    float* py = (float*)y->datas;
    float* px = (float*)x->datas;
    float* pw = (float*)w->datas;
    float* pb = NULL;

    conv_mode_t conv_mode = CONV_SIMPLE;
    float* pxcache = NULL;
    float* matw = NULL;
    float* matx = NULL;
    float* maty = NULL;

    float sum, v, weight;
    int ndim = x->ndim;
    int M = w->dims[0];
    int C = w->dims[1];
    int H = w->dims[2];
    int W = w->dims[3];
    int ch, i;

    if (nd->nin > 2) {
        b = nd->in[2];
        pb = (float*)b->datas;
    }
    if (ndim == 4) {
        int iC = x->dims[1];
        int iH = x->dims[2];
        int iW = x->dims[3];

        int oN = y->dims[0];
        int oC = w->dims[0];
        int oH = y->dims[2];
        int oW = y->dims[3];

        int MM = M / pdat->group;
        int CC = iC / pdat->group;

        typedef float(*pxtype)[iC][iH][iW];
        typedef float(*pwtype)[C][H][W];
        typedef float(*pytype)[M][oH][oW];
        typedef float(*pxcachetype)[(oC * pdat->group / M) * C][H][W];
        typedef float(*mwtype) /*[H * W * C]*/[MM];
        typedef float(*mxtype) /*[oH * oW]*/[H * W * C];
        typedef float(*mytype) /*[oH * oW]*/[MM];

        /* try im2col first */
        matw = malloc(MM * H * W * C * sizeof(float));
        matx = malloc(oH * oW * H * W * C * sizeof(float));
        maty = malloc(oH * oW * MM * sizeof(float));
        if (matw && matx && maty) {
            conv_mode = CONV_IM2COL;
        } else {
            if (matw) free(matw);
            if (matx) free(matx);
            if (maty) free(maty);

            /* then try cached conv */
            pxcache = malloc(oN * (oC * pdat->group / M) * C * H * W * sizeof(float));
            if (pxcache) {
                conv_mode = CONV_CACHED;
            }
        }

        if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED) {
            for (int h = 0; h < oH; ++h) {
                for (int w = 0; w < oW; ++w) {
                    int base_h = h * pdat->strides[0] - pdat->cpads[0];
                    int base_w = w * pdat->strides[1] - pdat->cpads[1];

                    if (pxcache) {
                        for (int n = 0; n < oN; ++n) {
                            for (int group_c = 0; group_c < oC * pdat->group / M; ++group_c) {
                                int base_c = group_c * C;
                                for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
                                    int input_h = base_h + i * pdat->dilations[0];
                                    if (input_h >= iH)
                                        break;
                                    for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
                                        int input_w = base_w + j * pdat->dilations[1];
                                        if (input_w >= iW)
                                            break;
                                        for (int w_channel = 0; w_channel < C; ++w_channel) {
                                            ch = base_c + w_channel;
                                            ((pxcachetype)pxcache)[n][ch][i][j] = ((pxtype)px)[n][ch][input_h][input_w];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int n = 0; n < oN; ++n) {
                        for (int c = 0; c < oC; ++c) {
                            int base_c = (c * pdat->group / M) * C;
                            sum = 0;
                            for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
                                int input_h = base_h + i * pdat->dilations[0];
                                if (input_h >= iH)
                                    break;
                                for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
                                    int input_w = base_w + j * pdat->dilations[1];
                                    if (input_w >= iW)
                                        break;
                                    for (int w_channel = 0; w_channel < C; ++w_channel) {
                                        ch = base_c + w_channel;
                                        if (pxcache) {
                                            v = ((pxcachetype)pxcache)[n][ch][i][j];
                                        } else {
                                            v = ((pxtype)px)[n][ch][input_h][input_w];
                                        }
                                        weight = ((pwtype)pw)[c][w_channel][i][j];
                                        sum += v * weight;
                                    }
                                }
                            }
                            if (pb)
                                sum += pb[c];
                            ((pytype)py)[n][c][h][w] = sum;
                        }
                    }
                }
            }
            if (pxcache) {
                free(pxcache);
            }
        } else if (conv_mode == CONV_IM2COL) {
            for (int g = 0; g < pdat->group; g++) {
                for (size_t m = 0; m < MM; m++) {
                    for (size_t c = 0; c < C; c++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                ((mwtype)matw)[c * H * W + h * W + w][m] = ((pwtype)pw)[g * MM + m][c][h][w];
                            }
                        }
                    }
                }

                for (int n = 0; n < oN; n++) {
                    for (size_t hh = 0; hh < oH; hh++) {
                        for (size_t ww = 0; ww < oW; ww++) {
                            int base_h = hh * pdat->strides[0] - pdat->cpads[0];
                            int base_w = ww * pdat->strides[1] - pdat->cpads[1];
                            for (size_t c = 0; c < C; c++) {
                                for (size_t h = 0; h < H; h++) {
                                    for (size_t w = 0; w < W; w++) {
                                        int ih = base_h + h * pdat->dilations[0];
                                        int iw = base_w + w * pdat->dilations[1];
                                        if (ih < 0 || iw < 0 || ih >= iH || iw >= iW) {
                                            ((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = 0.;
                                        } else {
                                            ((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = ((pxtype)px)[n][g * CC + c][ih][iw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    dgemm_float32(oH * oW, MM, H * W * C, matx, matw, maty);
                    for (int m = 0; m < MM; ++m) {
                        for (int h = 0; h < oH; ++h) {
                            for (int w = 0; w < oW; ++w) {
                                float t = ((mytype)maty)[h * oW + w][m];
                                if (pb) {
                                    t += pb[g * MM + m];
                                }
                                ((pytype)py)[n][g * MM + m][h][w] = t;
                            }
                        }
                    }
                }
            }
            free(matw);
            free(matx);
            free(maty);
        } else {
            /* never */
        }
    } else {
        int i_dim[ndim];
        int o_dim[ndim];
        int w_dim[ndim];
        int b_dim[ndim];

        memset(o_dim, 0, sizeof(o_dim));
        do {
            b_dim[0] = o_dim[0];
            for (i = 2; i < ndim; i++)
                b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
            sum = 0;
            memset(w_dim, 0, sizeof(w_dim));
            w_dim[0] = o_dim[1];
            do {
                if (w_dim[1] == 1)
                    break;
                i_dim[0] = b_dim[0];
                for (i = 2; i < ndim; i++)
                    i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
                for (ch = 0; ch < C; ch++) {
                    i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
                    w_dim[1] = ch;
                    for (i = 0; i < ndim; i++) {
                        if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                            v = 0;
                            break;
                        }
                    }
                    if (i >= ndim)
                        v = px[dim_offset(ndim, i_dim, x->dims)];
                    for (i = 0; i < ndim; i++) {
                        if ((w_dim[i] < 0) || (w_dim[i] >= w->dims[i])) {
                            weight = 0;
                            break;
                        }
                    }
                    if (i >= ndim)
                        weight = pw[dim_offset(ndim, w_dim, w->dims)];
                    sum += v * weight;
                }
                w_dim[1] = 0;
            } while (dim_next(ndim, w_dim, w->dims));
            if (pb)
                sum += pb[o_dim[1]];
            py[dim_offset(ndim, o_dim, y->dims)] = sum;
        } while (dim_next(ndim, o_dim, y->dims));
    }
}

static void Conv_forward_float64(node_t *nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    tensor_t* w = nd->in[1];
    tensor_t* b = NULL;
    double* py = (double*)y->datas;
    double* px = (double*)x->datas;
    double* pw = (double*)w->datas;
    double* pb = NULL;

    conv_mode_t conv_mode = CONV_SIMPLE;
    double* pxcache = NULL;
    double* matw = NULL;
    double* matx = NULL;
    double* maty = NULL;

    double sum, v, weight;
    int ndim = x->ndim;
    int M = w->dims[0];
    int C = w->dims[1];
    int H = w->dims[2];
    int W = w->dims[3];
    int ch, i;

    if (nd->nin > 2) {
        b = nd->in[2];
        pb = (double*)b->datas;
    }
    if (ndim == 4) {
        int iC = x->dims[1];
        int iH = x->dims[2];
        int iW = x->dims[3];

        int oN = y->dims[0];
        int oC = w->dims[0];
        int oH = y->dims[2];
        int oW = y->dims[3];

        int MM = M / pdat->group;
        int CC = iC / pdat->group;

        typedef double(*pxtype)[iC][iH][iW];
        typedef double(*pwtype)[C][H][W];
        typedef double(*pytype)[M][oH][oW];
        typedef double(*pxcachetype)[(oC * pdat->group / M) * C][H][W];
        typedef double(*mwtype) /*[H * W * C]*/[MM];
        typedef double(*mxtype) /*[oH * oW]*/[H * W * C];
        typedef double(*mytype) /*[oH * oW]*/[MM];

        /* try im2col first */
        matw = malloc(MM * H * W * C * sizeof(double));
        matx = malloc(oH * oW * H * W * C * sizeof(double));
        maty = malloc(oH * oW * MM * sizeof(double));
        if (matw && matx && maty) {
            conv_mode = CONV_IM2COL;
        } else {
            if (matw) free(matw);
            if (matx) free(matx);
            if (maty) free(maty);

            /* then try cached conv */
            pxcache = malloc(oN * (oC * pdat->group / M) * C * H * W * sizeof(double));
            if (pxcache) {
                conv_mode = CONV_CACHED;
            }
        }

        if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED) {
            for (int h = 0; h < oH; ++h) {
                for (int w = 0; w < oW; ++w) {
                    int base_h = h * pdat->strides[0] - pdat->cpads[0];
                    int base_w = w * pdat->strides[1] - pdat->cpads[1];

                    if (pxcache) {
                        for (int n = 0; n < oN; ++n) {
                            for (int group_c = 0; group_c < oC * pdat->group / M; ++group_c) {
                                int base_c = group_c * C;
                                for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
                                    int input_h = base_h + i * pdat->dilations[0];
                                    if (input_h >= iH)
                                        break;
                                    for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
                                        int input_w = base_w + j * pdat->dilations[1];
                                        if (input_w >= iW)
                                            break;
                                        for (int w_channel = 0; w_channel < C; ++w_channel) {
                                            ch = base_c + w_channel;
                                            ((pxcachetype)pxcache)[n][ch][i][j] = ((pxtype)px)[n][ch][input_h][input_w];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int n = 0; n < oN; ++n) {
                        for (int c = 0; c < oC; ++c) {
                            int base_c = (c * pdat->group / M) * C;
                            sum = 0;
                            for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
                                int input_h = base_h + i * pdat->dilations[0];
                                if (input_h >= iH)
                                    break;
                                for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
                                    int input_w = base_w + j * pdat->dilations[1];
                                    if (input_w >= iW)
                                        break;
                                    for (int w_channel = 0; w_channel < C; ++w_channel) {
                                        ch = base_c + w_channel;
                                        if (pxcache) {
                                            v = ((pxcachetype)pxcache)[n][ch][i][j];
                                        } else {
                                            v = ((pxtype)px)[n][ch][input_h][input_w];
                                        }
                                        weight = ((pwtype)pw)[c][w_channel][i][j];
                                        sum += v * weight;
                                    }
                                }
                            }
                            if (pb)
                                sum += pb[c];
                            ((pytype)py)[n][c][h][w] = sum;
                        }
                    }
                }
            }
            if (pxcache) {
                free(pxcache);
            }
        } else if (conv_mode == CONV_IM2COL) {
            for (int g = 0; g < pdat->group; g++) {
                for (size_t m = 0; m < MM; m++) {
                    for (size_t c = 0; c < C; c++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                ((mwtype)matw)[c * H * W + h * W + w][m] = ((pwtype)pw)[g * MM + m][c][h][w];
                            }
                        }
                    }
                }

                for (int n = 0; n < oN; n++) {
                    for (size_t hh = 0; hh < oH; hh++) {
                        for (size_t ww = 0; ww < oW; ww++) {
                            int base_h = hh * pdat->strides[0] - pdat->cpads[0];
                            int base_w = ww * pdat->strides[1] - pdat->cpads[1];
                            for (size_t c = 0; c < C; c++) {
                                for (size_t h = 0; h < H; h++) {
                                    for (size_t w = 0; w < W; w++) {
                                        int ih = base_h + h * pdat->dilations[0];
                                        int iw = base_w + w * pdat->dilations[1];
                                        if (ih < 0 || iw < 0 || ih >= iH || iw >= iW) {
                                            ((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = 0.;
                                        } else {
                                            ((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = ((pxtype)px)[n][g * CC + c][ih][iw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    dgemm_float64(oH * oW, MM, H * W * C, matx, matw, maty);
                    for (int m = 0; m < MM; ++m) {
                        for (int h = 0; h < oH; ++h) {
                            for (int w = 0; w < oW; ++w) {
                                float t = ((mytype)maty)[h * oW + w][m];
                                if (pb) {
                                    t += pb[g * MM + m];
                                }
                                ((pytype)py)[n][g * MM + m][h][w] = t;
                            }
                        }
                    }
                }
            }
            free(matw);
            free(matx);
            free(maty);
        } else {
            /* never */
        }
    } else {
        int i_dim[ndim];
        int o_dim[ndim];
        int w_dim[ndim];
        int b_dim[ndim];

        memset(o_dim, 0, sizeof(o_dim));
        do {
            b_dim[0] = o_dim[0];
            for (i = 2; i < ndim; i++)
                b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
            sum = 0;
            memset(w_dim, 0, sizeof(w_dim));
            w_dim[0] = o_dim[1];
            do {
                if (w_dim[1] == 1)
                    break;
                i_dim[0] = b_dim[0];
                for (i = 2; i < ndim; i++)
                    i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
                for (ch = 0; ch < C; ch++) {
                    i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
                    w_dim[1] = ch;
                    for (i = 0; i < ndim; i++) {
                        if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                            v = 0;
                            break;
                        }
                    }
                    if (i >= ndim)
                        v = px[dim_offset(ndim, i_dim, x->dims)];
                    for (i = 0; i < ndim; i++) {
                        if ((w_dim[i] < 0) || (w_dim[i] >= w->dims[i])) {
                            weight = 0;
                            break;
                        }
                    }
                    if (i >= ndim)
                        weight = pw[dim_offset(ndim, w_dim, w->dims)];
                    sum += v * weight;
                }
                w_dim[1] = 0;
            } while (dim_next(ndim, w_dim, w->dims));
            if (pb)
                sum += pb[o_dim[1]];
            py[dim_offset(ndim, o_dim, y->dims)] = sum;
        } while (dim_next(ndim, o_dim, y->dims));
    }
}

static void Conv_backward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *dy = nd->out[0]->grad;  // 输出的梯度 (dL/dy)
    tensor_t *x = nd->in[0];          // 输入张量 (x)
    tensor_t *w = nd->in[1];          // 卷积核 (w)
    tensor_t *dx = nd->in[0]->grad;   // 输入的梯度 (dL/dx)
    tensor_t *dw = nd->in[1]->grad;   // 卷积核的梯度 (dL/dw)
    tensor_t *b = (nd->nin > 2) ? nd->in[2] : NULL;  // 偏置项 (b)
    tensor_t *db = (b) ? b->grad : NULL;             // 偏置项的梯度 (dL/db)

    float *px = (float *)x->datas;
    float *pw = (float *)w->datas;
    float *pdy = (float *)dy->datas;
    float *pdx = (float *)dx->datas;
    float *pdw = (float *)dw->datas;
    // float *pb = (b) ? (float *)b->datas : NULL;
    float *pdb = (db) ? (float *)db->datas : NULL;

    // int M = w->dims[0];  // 输出通道数
    int C = w->dims[1];  // 输入通道数
    int H = w->dims[2];  // 卷积核高度
    int W = w->dims[3];  // 卷积核宽度

    int oN = dy->dims[0];  // 输出张量的批大小
    int oC = dy->dims[1];  // 输出张量的通道数
    int oH = dy->dims[2];  // 输出张量的高度
    int oW = dy->dims[3];  // 输出张量的宽度

    int iC = x->dims[1];  // 输入张量的通道数
    int iH = x->dims[2];  // 输入张量的高度
    int iW = x->dims[3];  // 输入张量的宽度

    int strideH = pdat->strides[0];  // 高度方向的步长
    int strideW = pdat->strides[1];  // 宽度方向的步长
    int padH = pdat->cpads[0];       // 高度方向的填充
    int padW = pdat->cpads[1];       // 宽度方向的填充
    int dilationH = pdat->dilations[0];  // 高度方向的膨胀
    int dilationW = pdat->dilations[1];  // 宽度方向的膨胀

    // 反向传播计算卷积核的梯度 dL/dw
    for (int n = 0; n < oN; ++n) {
        for (int c = 0; c < oC; ++c) {
            for (int h = 0; h < oH; ++h) {
                for (int w = 0; w < oW; ++w) {
                    float grad_output_value = pdy[n * oC * oH * oW + c * oH * oW + h * oW + w];

                    // 计算偏置梯度 dL/db
                    if (db) {
                        pdb[c] += grad_output_value;
                    }

                    // 计算卷积核梯度 dL/dw
                    for (int kh = 0; kh < H; ++kh) {
                        for (int kw = 0; kw < W; ++kw) {
                            for (int kc = 0; kc < C; ++kc) {
                                int ih = h * strideH - padH + kh * dilationH;
                                int iw = w * strideW - padW + kw * dilationW;
                                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                                    int input_index = n * iC * iH * iW + kc * iH * iW + ih * iW + iw;
                                    int kernel_index = c * C * H * W + kc * H * W + kh * W + kw;
                                    pdw[kernel_index] += px[input_index] * grad_output_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 反向传播计算输入的梯度 dL/dx
    for (int n = 0; n < oN; ++n) {
        for (int c = 0; c < oC; ++c) {
            for (int h = 0; h < oH; ++h) {
                for (int w = 0; w < oW; ++w) {
                    float grad_output_value = pdy[n * oC * oH * oW + c * oH * oW + h * oW + w];

                    // 反卷积传播梯度到输入
                    for (int kh = 0; kh < H; ++kh) {
                        for (int kw = 0; kw < W; ++kw) {
                            for (int kc = 0; kc < C; ++kc) {
                                int ih = h * strideH - padH + kh * dilationH;
                                int iw = w * strideW - padW + kw * dilationW;
                                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                                    int input_index = n * iC * iH * iW + kc * iH * iW + ih * iW + iw;
                                    int kernel_index = c * C * H * W + kc * H * W + kh * W + kw;
                                    pdx[input_index] += pw[kernel_index] * grad_output_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void Conv_init(node_t *nd) {
    if (!nd || !nd->in) {
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
        pdat->group = node_get_attr_int(nd, "group", 1);
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

void Conv_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    nd->in[1]->is_param = 1;
    if(nd->nin > 2) nd->in[2]->is_param = 1;
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];
    tensor_t* x = nd->in[0];
    tensor_t* w = nd->in[1];
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
    dims[1] = w->dims[0];
    for (i = 0; i < ndim - 2; i++) {
        switch (pdat->auto_pad) {
            case AUTO_PAD_NOTSET:
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

void Conv_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            Conv_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Conv_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Conv_forward_float64(nd);
            break;
        default:
            break;
    }
}

void Conv_backward(node_t *nd) {
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
        case TENSOR_TYPE_FLOAT16:
            // Conv_backward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Conv_backward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            // Conv_backward_float64(nd);
            break;
        default:
            break;
    }
}

void Conv_exit(node_t *nd) {
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

void op_Conv_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Conv_init;
    nd->op->reshape     = Conv_reshape;
    nd->op->forward     = Conv_forward;
    nd->op->backward    = Conv_backward;
    nd->op->exit        = Conv_exit;
}