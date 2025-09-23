#include <evo/dev/cpu/kernel.h>
#include <math.h>

// ==================================================================================== //
//                                    symmetric quantize
// ==================================================================================== //


#define MAX(a, b) ((a) > (b) ? (a) : (b))

void Quant_symmetric_float32_to_int16_cpu(float *input, int16_t *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = (int16_t)roundf(input[i] / scale);
    }
}

void Dequant_symmetric_int16_to_float32_cpu(int16_t *input, float *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}

void Quant_symmetric_float32_to_int8_cpu(float *input, int8_t *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = (int8_t)roundf(input[i] / scale);
    }
}

void Dequant_symmetric_int8_to_float32_cpu(int8_t *input, float *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}

void Quant_symmetric_float32_to_uint8_cpu(float *input, uint8_t *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        int32_t quantized_val = (int32_t)roundf(input[i] / scale);
        // Clamp to uint8 range [0, 255] since uint8 cannot represent negative values
        output[i] = (uint8_t)(quantized_val < 0 ? 0 : (quantized_val > 255 ? 255 : quantized_val));
    }
}

void Dequant_symmetric_uint8_to_float32_cpu(uint8_t *input, float *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}


// ==================================================================================== //
//                                    asymmetric quantize
// ==================================================================================== //


void Quant_asymmetric_float32_to_int16_cpu(float *input, int16_t *output, int size, float scale, int16_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (int16_t)roundf(input[i] / scale) + zero_point;
    }
}

void Dequant_asymmetric_int16_to_float32_cpu(int16_t *input, float *output, int size, float scale, int16_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - zero_point) * scale;
    }
}

void Quant_asymmetric_float32_to_int8_cpu(float *input, int8_t *output, int size, float scale, int8_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (int8_t)roundf(input[i] / scale) + zero_point;
    }
}

void Dequant_asymmetric_int8_to_float32_cpu(int8_t *input, float *output, int size, float scale, int8_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - zero_point) * scale;
    }
}

void Quant_asymmetric_float32_to_uint8_cpu(float *input, uint8_t *output, int size, float scale, uint8_t zero_point) {
    for (int i = 0; i < size; i++) {
        int32_t quantized_val = (int32_t)roundf(input[i] / scale) + zero_point;
        // Clamp to uint8 range
        output[i] = (uint8_t)(quantized_val < 0 ? 0 : (quantized_val > 255 ? 255 : quantized_val));
    }
}

void Dequant_asymmetric_uint8_to_float32_cpu(uint8_t *input, float *output, int size, float scale, uint8_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - zero_point) * scale;
    }
}


void Quant_asymmetric_float32_to_int32_cpu(float *input, int32_t *output, int size, float scale, int32_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (int32_t)roundf(input[i] / scale) + zero_point;
    }
}

// ==================================================================================== //
//                                    dynamic range quantize
// ==================================================================================== //

// 动态范围量化 (int16_t)
void Quant_dynrange_float32_to_int16_cpu(float *input, int16_t *output, int size) {
    float min = input[0], max = input[0];
    
    // 寻找最小值和最大值
    for (int i = 1; i < size; i++) {
        if (input[i] < min) min = input[i];
        if (input[i] > max) max = input[i];
    }

    // 计算 scale
    float scale = (max - min) / (65535.0); // 映射到 int16_t 范围 [-32768, 32767]

    // 量化
    for (int i = 0; i < size; i++) {
        output[i] = (int16_t)((input[i] - min) / scale) - 32768;
    }
}

// 动态范围反量化 (int16_t)
void Dequant_dynrange_int16_to_float32_cpu(int16_t *input, float *output, int size, float min, float max) {
    float scale = (max - min) / 65535.0;
    for (int i = 0; i < size; i++) {
        output[i] = ((input[i] + 32768) * scale) + min;
    }
}


// 动态范围量化 (int8_t)
void Quant_dynrange_float32_to_int8_cpu(float *input, int8_t *output, int size) {
    float min = input[0], max = input[0];
    
    // 寻找最小值和最大值
    for (int i = 1; i < size; i++) {
        if (input[i] < min) min = input[i];
        if (input[i] > max) max = input[i];
    }

    // 计算 scale
    float scale = (max - min) / (255.0); // 映射到 int8_t 范围 [-128, 127]

    // 量化
    for (int i = 0; i < size; i++) {
        output[i] = (int8_t)((input[i] - min) / scale) - 128;
    }
}

// 动态范围反量化 (int8_t)
void Dequant_dynrange_int8_to_float32_cpu(int8_t *input, float *output, int size, float min, float max) {
    float scale = (max - min) / 255.0;
    for (int i = 0; i < size; i++) {
        output[i] = ((input[i] + 128) * scale) + min;
    }
}

// ==================================================================================== //
//                                    log quantize
// ==================================================================================== //


// 对数量化 (int16_t)
void Quant_log_float32_to_int16_cpu(float *input, int16_t *output, int size) {
    for (int i = 0; i < size; i++) {
        // 量化使用对数
        output[i] = (int16_t)(roundf(logf(input[i] + 1e-9) * 1000)); // 缩放值使其符合 int16_t 范围
    }
}

// 对数量化反量化 (int16_t)
void Dequant_log_int16_to_float32_cpu(int16_t *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        // 反量化使用指数函数
        output[i] = expf(input[i] / 1000.0);
    }
}

// 对数量化 (int8_t)
void Quant_log_float32_to_int8_cpu(float *input, int8_t *output, int size) {
    for (int i = 0; i < size; i++) {
        // 量化使用对数
        output[i] = (int8_t)(roundf(logf(input[i] + 1e-9) * 100)); // 缩放值使其符合 int8_t 范围
    }
}

// 对数量化反量化 (int8_t)
void Dequant_log_int8_to_float32_cpu(int8_t *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        // 反量化使用指数函数
        output[i] = expf(input[i] / 100.0);
    }
}



void Quant_asymmetric_float32_to_int_cpu(float *input, void *output, int size, float scale, int32_t zero_point, int width, int sign) {
    // Check for invalid width
    if (width <= 0 || width > 32) {
        return;
    }

    // Compute scaling factors based on bit-width and signed/unsigned configuration
    int max_value = (1 << (width - 1)) - 1;  // max value for signed
    int min_value = -(1 << (width - 1));     // min value for signed
    unsigned int max_u_value = (1 << width) - 1;  // max value for unsigned

    if (sign) {
        // Signed quantization
        if (width <= 8) {
            int8_t *out = (int8_t *)output;
            for (int i = 0; i < size; i++) {
                // Perform quantization without math.h
                int32_t quantized_value = (int32_t)(input[i] / scale) + zero_point;
                // Clamp to the signed range
                if (quantized_value > max_value) {
                    out[i] = max_value;
                } else if (quantized_value < min_value) {
                    out[i] = min_value;
                } else {
                    out[i] = (int8_t)quantized_value;
                }
            }
        } else if (width <= 16) {
            int16_t *out = (int16_t *)output;
            for (int i = 0; i < size; i++) {
                // Perform quantization without math.h
                int32_t quantized_value = (int32_t)(input[i] / scale) + zero_point;
                // Clamp to the signed range
                if (quantized_value > max_value) {
                    out[i] = max_value;
                } else if (quantized_value < min_value) {
                    out[i] = min_value;
                } else {
                    out[i] = (int16_t)quantized_value;
                }
            }
        } else if (width <= 32) {
            int32_t *out = (int32_t *)output;
            for (int i = 0; i < size; i++) {
                // Perform quantization without math.h
                int32_t quantized_value = (int32_t)(input[i] / scale) + zero_point;
                // Clamp to the signed range
                if (quantized_value > max_value) {
                    out[i] = max_value;
                } else if (quantized_value < min_value) {
                    out[i] = min_value;
                } else {
                    out[i] = quantized_value;
                }
            }
        }
    } else {
        // Unsigned quantization
        if (width <= 8) {
            uint8_t *out = (uint8_t *)output;
            for (int i = 0; i < size; i++) {
                // Perform quantization without math.h
                int32_t quantized_value = (int32_t)(input[i] / scale) + zero_point;
                // Clamp to the unsigned range [0, 255]
                if (quantized_value > max_u_value) {
                    out[i] = max_u_value;
                } else if (quantized_value < 0) {
                    out[i] = 0;
                } else {
                    out[i] = (uint8_t)quantized_value;
                }
            }
        } else if (width <= 16) {
            uint16_t *out = (uint16_t *)output;
            for (int i = 0; i < size; i++) {
                // Perform quantization without math.h
                int32_t quantized_value = (int32_t)(input[i] / scale) + zero_point;
                // Clamp to the unsigned range [0, 65535]
                if (quantized_value > max_u_value) {
                    out[i] = max_u_value;
                } else if (quantized_value < 0) {
                    out[i] = 0;
                } else {
                    out[i] = (uint16_t)quantized_value;
                }
            }
        } else if (width <= 32) {
            uint32_t *out = (uint32_t *)output;
            for (int i = 0; i < size; i++) {
                // Perform quantization without math.h
                int32_t quantized_value = (int32_t)(input[i] / scale) + zero_point;
                // Clamp to the unsigned range [0, 4294967295]
                if (quantized_value > max_u_value) {
                    out[i] = max_u_value;
                } else if (quantized_value < 0) {
                    out[i] = 0;
                } else {
                    out[i] = (uint32_t)quantized_value;
                }
            }
        }
    }
}


void Dequant_asymmetric_int_to_float32_cpu(void *input, float *output, int size, float scale, int32_t zero_point, int width, int sign) {
    // Check for invalid width
    if (width <= 0 || width > 32) {
        return;
    }

    if (sign) {
        // Signed dequantization
        if (width <= 8) {
            int8_t *in = (int8_t *)input;
            for (int i = 0; i < size; i++) {
                // Reverse quantization (Dequantization) for signed values
                output[i] = (in[i] - zero_point) * scale;
            }
        } else if (width <= 16) {
            int16_t *in = (int16_t *)input;
            for (int i = 0; i < size; i++) {
                // Reverse quantization (Dequantization) for signed values
                output[i] = (in[i] - zero_point) * scale;
            }
        } else if (width <= 32) {
            int32_t *in = (int32_t *)input;
            for (int i = 0; i < size; i++) {
                // Reverse quantization (Dequantization) for signed values
                output[i] = (in[i] - zero_point) * scale;
            }
        }
    } else {
        // Unsigned dequantization
        if (width <= 8) {
            uint8_t *in = (uint8_t *)input;
            for (int i = 0; i < size; i++) {
                // Reverse quantization (Dequantization) for unsigned values
                output[i] = (in[i] - zero_point) * scale;
            }
        } else if (width <= 16) {
            uint16_t *in = (uint16_t *)input;
            for (int i = 0; i < size; i++) {
                // Reverse quantization (Dequantization) for unsigned values
                output[i] = (in[i] - zero_point) * scale;
            }
        } else if (width <= 32) {
            uint32_t *in = (uint32_t *)input;
            for (int i = 0; i < size; i++) {
                // Reverse quantization (Dequantization) for unsigned values
                output[i] = (in[i] - zero_point) * scale;
            }
        }
    }
}


// Function to calculate entropy for quantized values
float calculate_entropy(float* data, int size, int width, int sign) {
    // Define the number of bins for the histogram based on width
    int num_bins = (1 << width); // e.g., 256 bins for 8-bit, 65536 bins for 16-bit

    // Define the range based on sign
    int min_range = (sign == 0) ? 0 : -(1 << (width - 1));  // For signed, it's -max to +max, else it's 0 to max
    int max_range = (sign == 0) ? (1 << width) - 1 : (1 << (width - 1)) - 1;

    // Initialize the histogram
    float histogram[num_bins];
    for(int i = 0; i < num_bins; i++) {
        histogram[i] = 0;
    }

    // Map data to the appropriate quantization range
    for (int i = 0; i < size; i++) {
        // Scale data to match the quantization range
        int bin = (int)(data[i] * ((float)(num_bins - 1) / (max_range - min_range)) + 0.5f);

        // Clip the bin value to the valid range
        if (bin >= 0 && bin < num_bins) {
            histogram[bin]++;
        }
    }

    // Normalize the histogram to calculate probabilities
    for (int i = 0; i < num_bins; i++) {
        histogram[i] /= size;
    }

    // Calculate entropy based on the normalized histogram
    float entropy = 0.0f;
    for (int i = 0; i < num_bins; i++) {
        if (histogram[i] > 0) {
            entropy -= histogram[i] * logf(histogram[i]);
        }
    }
    return entropy;
}

void Quant_solve(float* data, int size, float* scale, int* zero_point, int width, int sign, int method) {
    if(!data) return;

    *scale = 0.f;
    *zero_point = 0;

    // Initialize the tensor data and find the min and max values
    float max = data[0], min = data[0];
    for (int i = 0; i < size; i++) {
        if (data[i] > max) max = data[i];
        if (data[i] < min) min = data[i];
    }

    int max_range = (1 << (width - 1)) - 1;  // Max value for signed quantization
    int min_range = -(1 << (width - 1));     // Min value for signed quantization

    if (sign == 0) {  // Unsigned quantization
        max_range = (1 << width) - 1;  // Max value for unsigned quantization
        min_range = 0;                 // Min value for unsigned quantization
    }


    float range = max - min;

    if (range == 0) {
        *scale = 1.0f;  
        *zero_point = 0;
        return;
    }

     // Choose the method based on the input
    if (method == 0) {
        // Min-Max symmetric quantization
        if (max < 0) max = -max;
        if (min < 0) min = -min;
        *scale = MAX(max, min) / (float)max_range;  // Scale for quantization
        *zero_point = 0; // No zero-point for symmetric quantization
    }  else if (method == 1) {
        // Min-Max asymmetric quantization
        if (max < 0) max = -max;
        if (min < 0) min = -min;
        *scale = (max - min) / (float)(max_range);  // Scale for quantization
        *zero_point = (int)roundf(-(min / *scale)); // Zero-point calculation
    } else if (method == 2) {       // Entropy-based quantization (symmetric)
        float entropy = calculate_entropy(data, size, width, sign);

        // Scale based on entropy
        *scale = (max - min) / (float)(max_range);  // Scale based on the full range
        *zero_point = 0; // Zero-point could be zero for entropy-based methods (depending on data distribution)

        // Example of how entropy could be used to adjust scale:
        *scale *= expf(-entropy / 10.0f);  // Adjust scale using entropy for better fitting (can be fine-tuned)
    } else if (method == 3) {       // Entropy-based asymmetric quantization
        float entropy = calculate_entropy(data, size, width, sign);

        // Scale based on entropy
        *scale = (max - min) / (float)(max_range);  // Scale calculation based on range
        *zero_point = (int)roundf(-(min / *scale)); // Zero-point calculation

        // Use entropy to adjust scale:
        *scale *= expf(-entropy / 10.0f);  // Scale adjustment using entropy
    }
}

#undef MAX