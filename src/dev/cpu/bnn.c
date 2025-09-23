#include <evo/dev/cpu/kernel.h>


// 64-bit binary_pack
void BinPack64_float32_cpu(float *activations, uint64_t *packed_output, int num_elements) {
    for (int i = 0; i < num_elements; i += 64) {
        uint64_t packed_val = 0;
        for (int j = 0; j < 64 && (i + j) < num_elements; j++) {
            // 将激活值>0映射为1，其他为0
            if (activations[i + j] > 0) {
                packed_val |= ((uint64_t)1 << j);
            }
        }
        packed_output[i / 64] = packed_val;
    }
}

// 32-bit binary_pack
void BinPack32_float32_cpu(float *activations, uint32_t *packed_output, int num_elements) {
    for (int i = 0; i < num_elements; i += 32) {
        uint32_t packed_val = 0;
        for (int j = 0; j < 32 && (i + j) < num_elements; j++) {
            // 将激活值>0映射为1，其他为0
            if (activations[i + j] > 0) {
                packed_val |= ((uint32_t)1 << j);
            }
        }
        packed_output[i / 32] = packed_val;
    }
}

// 64-bit binary_unpack
void BinUnpack64_float32_cpu(uint64_t *packed_input, float *unpacked_output, int num_elements) {
    for (int i = 0; i < num_elements; i += 64) {
        uint64_t packed_val = packed_input[i / 64];
        for (int j = 0; j < 64 && (i + j) < num_elements; j++) {
            // 解包并恢复浮点值 -1 或 +1
            unpacked_output[i + j] = (packed_val & ((uint64_t)1 << j)) ? 1.0f : -1.0f;
        }
    }
}

// 32-bit binary_unpack
void BinUnpack32_float32_cpu(uint32_t *packed_input, float *unpacked_output, int num_elements) {
    for (int i = 0; i < num_elements; i += 32) {
        uint32_t packed_val = packed_input[i / 32];
        for (int j = 0; j < 32 && (i + j) < num_elements; j++) {
            // 解包并恢复浮点值 -1 或 +1
            unpacked_output[i + j] = (packed_val & ((uint32_t)1 << j)) ? 1.0f : -1.0f;
        }
    }
}


// popcount for 64-bit integer
static inline int popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = x + (x >> 8);
    x = x + (x >> 16);
    x = x + (x >> 32);
    return x & 0x7F;
}

// popcount for 32-bit integer
static inline int popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
}


// 64-bit binary convolution forward
void BinConv64_forward_cpu(uint64_t *input, uint64_t *weight, int32_t *output,
                           int input_height, int input_width, 
                           int output_height, int output_width, 
                           int kernel_size, int stride, int padding, 
                           int input_channels, int output_channels) {
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int oy = 0; oy < output_height; ++oy) {
            for (int ox = 0; ox < output_width; ++ox) {
                int popcount_sum = 0;

                for (int ic = 0; ic < input_channels; ++ic) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int ix = ox * stride + kx - padding;
                            int iy = oy * stride + ky - padding;

                            if (ix >= 0 && ix < input_width && iy >= 0 && iy < input_height) {
                                int index = ic * input_height * input_width + iy * input_width + ix;
                                uint64_t xnor_result = ~(input[index] ^ weight[index]);
                                popcount_sum += popcount64(xnor_result);
                            }
                        }
                    }
                }

                int result = 2 * popcount_sum - kernel_size * kernel_size * input_channels;
                output[oc * output_height * output_width + oy * output_width + ox] = result;
            }
        }
    }
}

// 32-bit binary convolution forward
void BinConv32_forward_cpu(uint32_t *input, uint32_t *weight, int32_t *output,
                           int input_height, int input_width, 
                           int output_height, int output_width, 
                           int kernel_size, int stride, int padding, 
                           int input_channels, int output_channels) {
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int oy = 0; oy < output_height; ++oy) {
            for (int ox = 0; ox < output_width; ++ox) {
                int popcount_sum = 0;

                for (int ic = 0; ic < input_channels; ++ic) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int ix = ox * stride + kx - padding;
                            int iy = oy * stride + ky - padding;

                            if (ix >= 0 && ix < input_width && iy >= 0 && iy < input_height) {
                                int index = ic * input_height * input_width + iy * input_width + ix;
                                uint32_t xnor_result = ~(input[index] ^ weight[index]);
                                popcount_sum += popcount32(xnor_result);
                            }
                        }
                    }
                }

                int result = 2 * popcount_sum - kernel_size * kernel_size * input_channels;
                output[oc * output_height * output_width + oy * output_width + ox] = result;
            }
        }
    }
}


// Main float32 binary convolution function with bit width selection
void BinConv_float32_forward_cpu(float *input, float *weight, int32_t *output, 
                                 int input_height, int input_width, 
                                 int output_height, int output_width, 
                                 int kernel_size, int stride, int padding, 
                                 int input_channels, int output_channels, int bit_width) {

    int num_elements = input_height * input_width * input_channels;

    if (bit_width == 32) {
        uint32_t packed_input[(num_elements + 31) / 32 * sizeof(uint32_t)];
        uint32_t packed_weight[output_channels * input_channels * kernel_size * kernel_size * sizeof(uint32_t)];

        BinPack32_float32_cpu(input, packed_input, num_elements);
        BinPack32_float32_cpu(weight, packed_weight, output_channels * input_channels * kernel_size * kernel_size);

        BinConv32_forward_cpu(packed_input, packed_weight, output, input_height, input_width, 
                              output_height, output_width, kernel_size, stride, padding, 
                              input_channels, output_channels);

    } else if (bit_width == 64) {
        uint64_t packed_input[(num_elements + 63) / 64 * sizeof(uint64_t)];
        uint64_t packed_weight[output_channels * input_channels * kernel_size * kernel_size * sizeof(uint64_t)];

        BinPack64_float32_cpu(input, packed_input, num_elements);
        BinPack64_float32_cpu(weight, packed_weight, output_channels * input_channels * kernel_size * kernel_size);

        BinConv64_forward_cpu(packed_input, packed_weight, output, input_height, input_width, 
                              output_height, output_width, kernel_size, stride, padding, 
                              input_channels, output_channels);
    }
}

// 32-bit Binary GEMM forward
void BinGemm32_forward_cpu(uint32_t *input, uint32_t *weight, int32_t *output, 
                           int M, int N, int K) {
    // 遍历输出矩阵的每个元素
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int popcount_sum = 0;

            // 遍历 K 维度进行相似性计算
            for (int k = 0; k < (K + 31) / 32; ++k) {
                int input_idx = m * ((K + 31) / 32) + k;
                int weight_idx = n * ((K + 31) / 32) + k;

                // 计算 XNOR 结果
                uint32_t xnor_result = ~(input[input_idx] ^ weight[weight_idx]);

                // 累加 popcount 结果
                popcount_sum += popcount32(xnor_result);
            }

            // 将 popcount 结果转换为 -K 到 +K 的范围并写入输出
            int result = 2 * popcount_sum - K;
            output[m * N + n] = result;
        }
    }
}

// 64-bit Binary GEMM forward
void BinGemm64_forward_cpu(uint64_t *input, uint64_t *weight, int32_t *output, 
                           int M, int N, int K) {
    // 遍历输出矩阵的每个元素
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int popcount_sum = 0;

            // 遍历 K 维度进行相似性计算
            for (int k = 0; k < (K + 63) / 64; ++k) {
                int input_idx = m * ((K + 63) / 64) + k;
                int weight_idx = n * ((K + 63) / 64) + k;

                // 计算 XNOR 结果
                uint64_t xnor_result = ~(input[input_idx] ^ weight[weight_idx]);

                // 累加 popcount 结果
                popcount_sum += popcount64(xnor_result);
            }

            // 将 popcount 结果转换为 -K 到 +K 的范围并写入输出
            int result = 2 * popcount_sum - K;
            output[m * N + n] = result;
        }
    }
}


void BinGemm_float32_forward_cpu(float *input, float *weight, int32_t *output, 
                                 int M, int N, int K, int bit_width) {

    if (bit_width == 32) {
        uint32_t packed_input[(M * K + 31) / 32 * sizeof(uint32_t)];
        uint32_t packed_weight[(N * K + 31) / 32 * sizeof(uint32_t)];

        BinPack32_float32_cpu(input, packed_input, M * K);
        BinPack32_float32_cpu(weight, packed_weight, N * K);
        BinGemm32_forward_cpu(packed_input, packed_weight, output, M, N, K);

    } else if (bit_width == 64) {
        uint64_t packed_input[(M * K + 63) / 64 * sizeof(uint64_t)];
        uint64_t packed_weight[(N * K + 63) / 64 * sizeof(uint64_t)];

        BinPack64_float32_cpu(input, packed_input, M * K);
        BinPack64_float32_cpu(weight, packed_weight, N * K);
        BinGemm64_forward_cpu(packed_input, packed_weight, output, M, N, K);
    }
}