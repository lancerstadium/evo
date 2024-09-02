#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>

typedef struct {
    char* mode;
} operator_pdata_t;

static void Upsample_uint8(node_t* nd) {
    operator_pdata_t *pdat = (operator_pdata_t*)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];

    int batch_size = x->dims[0];
    int channels = x->dims[1];
    int input_height = x->dims[2];
    int input_width = x->dims[3];
    int output_height = y->dims[2];
    int output_width = y->dims[3];

    uint8_t* input_data = (uint8_t*)x->datas;
    uint8_t* output_data = (uint8_t*)y->datas;

    if (strcmp(pdat->mode, "nearest") == 0) {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        int in_h = (int)(h * input_height / output_height);
                        int in_w = (int)(w * input_width / output_width);
                        output_data[b * channels * output_height * output_width +
                                    c * output_height * output_width +
                                    h * output_width + w] =
                            input_data[b * channels * input_height * input_width +
                                       c * input_height * input_width +
                                       in_h * input_width + in_w];
                    }
                }
            }
        }
    } else if (strcmp(pdat->mode, "bilinear") == 0) {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        float in_h = h * (input_height - 1) / (float)(output_height - 1);
                        float in_w = w * (input_width - 1) / (float)(output_width - 1);

                        int h0 = (int)in_h;
                        int w0 = (int)in_w;
                        int h1 = h0 + 1;
                        int w1 = w0 + 1;

                        if (h1 >= input_height) h1 = input_height - 1;
                        if (w1 >= input_width) w1 = input_width - 1;

                        float h_weight = in_h - h0;
                        float w_weight = in_w - w0;

                        float top = (1 - w_weight) * input_data[b * channels * input_height * input_width +
                                                                c * input_height * input_width +
                                                                h0 * input_width + w0] +
                                    w_weight * input_data[b * channels * input_height * input_width +
                                                          c * input_height * input_width +
                                                          h0 * input_width + w1];
                        float bottom = (1 - w_weight) * input_data[b * channels * input_height * input_width +
                                                                   c * input_height * input_width +
                                                                   h1 * input_width + w0] +
                                       w_weight * input_data[b * channels * input_height * input_width +
                                                             c * input_height * input_width +
                                                             h1 * input_width + w1];

                        output_data[b * channels * output_height * output_width +
                                    c * output_height * output_width +
                                    h * output_width + w] =
                            (uint8_t)((1 - h_weight) * top + h_weight * bottom + 0.5f);  // 四舍五入
                    }
                }
            }
        }
    }
}

static void Upsample_float32(node_t* nd) {
    fprintf(stderr, "Upsample_float32\n");
    operator_pdata_t *pdat = (operator_pdata_t*)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];

    int batch_size = x->dims[0];
    int channels = x->dims[1];
    int input_height = x->dims[2];
    int input_width = x->dims[3];
    int output_height = y->dims[2];
    int output_width = y->dims[3];

    float* input_data = (float*)x->datas;
    float* output_data = (float*)y->datas;

    if (strcmp(pdat->mode, "nearest") == 0) {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        int in_h = (int)(h * input_height / output_height);
                        int in_w = (int)(w * input_width / output_width);
                        output_data[b * channels * output_height * output_width +
                                    c * output_height * output_width +
                                    h * output_width + w] =
                            input_data[b * channels * input_height * input_width +
                                       c * input_height * input_width +
                                       in_h * input_width + in_w];
                    }
                }
            }
        }
    } else if (strcmp(pdat->mode, "bilinear") == 0) {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        float in_h = h * (input_height - 1) / (float)(output_height - 1);
                        float in_w = w * (input_width - 1) / (float)(output_width - 1);

                        int h0 = (int)in_h;
                        int w0 = (int)in_w;
                        int h1 = h0 + 1;
                        int w1 = w0 + 1;

                        if (h1 >= input_height) h1 = input_height - 1;
                        if (w1 >= input_width) w1 = input_width - 1;

                        float h_weight = in_h - h0;
                        float w_weight = in_w - w0;

                        float top = (1 - w_weight) * input_data[b * channels * input_height * input_width +
                                                                c * input_height * input_width +
                                                                h0 * input_width + w0] +
                                    w_weight * input_data[b * channels * input_height * input_width +
                                                          c * input_height * input_width +
                                                          h0 * input_width + w1];
                        float bottom = (1 - w_weight) * input_data[b * channels * input_height * input_width +
                                                                   c * input_height * input_width +
                                                                   h1 * input_width + w0] +
                                       w_weight * input_data[b * channels * input_height * input_width +
                                                             c * input_height * input_width +
                                                             h1 * input_width + w1];

                        output_data[b * channels * output_height * output_width +
                                    c * output_height * output_width +
                                    h * output_width + w] =
                            (1 - h_weight) * top + h_weight * bottom;
                    }
                }
            }
        }
    }
}


void op_Upsample_dft(node_t* nd) {
    // 1. Upsample init
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 2) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) || (nd->in[1]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED || nd->in[1]->type != TENSOR_TYPE_FLOAT32) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->mode = node_get_attr_string(nd, "mode", "nearest");
        nd->priv = pdat;
    }
    // 2. Upsample reshape
    tensor_t *x = nd->in[0];
    tensor_t *sc = nd->in[1];
    tensor_t *y = nd->out[0];
    y->type = x->type;
    int new_dims[x->ndim];
    float* p = (float*)sc->datas;
    for (int i = 0; i < x->ndim; i++) {
        new_dims[i] = x->dims[i] * (p[i]);
    }
    tensor_reshape(y, x->ndim, new_dims);
    
    // 3. Upsample run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_UINT8:
            Upsample_uint8(nd);
        case TENSOR_TYPE_FLOAT32:
            Upsample_float32(nd);
            break;
        default:
            break;
    }
    // 4. Upsample exit
    if (pdat) {
        free(pdat);
    }
    nd->priv = NULL;
    return;
}