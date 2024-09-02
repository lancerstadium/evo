#include <evo/resolver.h>
#include <evo/util/math.h>


typedef struct {
    char* mode;
} operator_pdata_t;


void upsample_nearest(float *input, float *output, int channels, int input_height, int input_width, int scale_factor) {
    int output_height = input_height * scale_factor;
    int output_width = input_width * scale_factor;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                // Nearest neighbor interpolation
                int input_h = h / scale_factor;
                int input_w = w / scale_factor;

                output[c * output_height * output_width + h * output_width + w] =
                    input[c * input_height * input_width + input_h * input_width + input_w];
            }
        }
    }
}

void upsample_bilinear(float *input, float *output, int channels, int input_height, int input_width, int scale_factor) {
    int output_height = input_height * scale_factor;
    int output_width = input_width * scale_factor;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                // Compute the corresponding position in the input image
                float in_h = h / (float)scale_factor;
                float in_w = w / (float)scale_factor;

                int h0 = (int)in_h;
                int w0 = (int)in_w;
                int h1 = h0 + 1;
                int w1 = w0 + 1;

                // Boundary conditions
                if (h1 >= input_height) h1 = input_height - 1;
                if (w1 >= input_width) w1 = input_width - 1;

                float h_weight = in_h - h0;
                float w_weight = in_w - w0;

                // Bilinear interpolation
                float top = (1 - w_weight) * input[c * input_height * input_width + h0 * input_width + w0] +
                            w_weight * input[c * input_height * input_width + h0 * input_width + w1];
                float bottom = (1 - w_weight) * input[c * input_height * input_width + h1 * input_width + w0] +
                               w_weight * input[c * input_height * input_width + h1 * input_width + w1];

                output[c * output_height * output_width + h * output_width + w] =
                    (1 - h_weight) * top + h_weight * bottom;
            }
        }
    }
}


static void Upsample_float32(node_t* nd) {


}


void op_Upsample_dft(node_t* nd) {

    // 3.Upsample run
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT32:
            Upsample_float32(nd);
            break;
        default:
            break;
    }
}