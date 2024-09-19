#include <evo/util/math.h>
#include <stdlib.h>
#include <math.h>

int imin(int a, int b) {
    return a <= b ? a : b;
}

int imax(int a, int b) {
    return a >= b ? a : b;
}

int min_abs(int a, int b) {
    return imin(abs(a), abs(b));
}

int max_abs(int a, int b) {
    return imax(abs(a), abs(b));
}

static int solve_gcd(int large, int small) {
    int val = large % small;
    return 0 == val ? small : gcd(small, val);
}

int gcd(int a, int b) {
    if (0 == a || 0 == b)
        return 0;

    return solve_gcd(max_abs(a, b), min_abs(a, b));
}

int lcm(int a, int b) {
    if (0 == a || 0 == b)
        return 0;

    return abs(a * b) / solve_gcd(max_abs(a, b), min_abs(a, b));
}

int align(int value, int step) {
    const int mask = ~(abs(step) - 1);
    return (value + step) & mask;
}

void* align_address(void* address, int step) {
    const size_t mask = ~(abs(step) - 1);
    return (void*)((size_t)address & mask);
}


// ==================================================================================== //
//                                    symmetric quantize
// ==================================================================================== //

void symmetric_quantize_float32_to_int16(float *input, int16_t *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = (int16_t)roundf(input[i] / scale);
    }
}

void symmetric_dequantize_int16_to_float32(int16_t *input, float *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}

void symmetric_quantize_float32_to_int8(float *input, int8_t *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = (int8_t)roundf(input[i] / scale);
    }
}

void symmetric_dequantize_int8_to_float32(int8_t *input, float *output, int size, float scale) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}

// ==================================================================================== //
//                                    asymmetric quantize
// ==================================================================================== //


void asymmetric_quantize_float32_to_int16(float *input, int16_t *output, int size, float scale, int16_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (int16_t)roundf(input[i] / scale) + zero_point;
    }
}

void asymmetric_dequantize_int16_to_float32(int16_t *input, float *output, int size, float scale, int16_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - zero_point) * scale;
    }
}

void asymmetric_quantize_float32_to_int8(float *input, int8_t *output, int size, float scale, int8_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (int8_t)roundf(input[i] / scale) + zero_point;
    }
}

void asymmetric_dequantize_int8_to_float32(int8_t *input, float *output, int size, float scale, int8_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - zero_point) * scale;
    }
}


// ==================================================================================== //
//                                    dynamic range quantize
// ==================================================================================== //

// 动态范围量化 (int16_t)
void dynamic_range_quantize_float32_to_int16(float *input, int16_t *output, int size) {
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
void dynamic_range_dequantize_int16_to_float32(int16_t *input, float *output, int size, float min, float max) {
    float scale = (max - min) / 65535.0;
    for (int i = 0; i < size; i++) {
        output[i] = ((input[i] + 32768) * scale) + min;
    }
}


// 动态范围量化 (int8_t)
void dynamic_range_quantize_float32_to_int8(float *input, int8_t *output, int size) {
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
void dynamic_range_dequantize_int8_to_float32(int8_t *input, float *output, int size, float min, float max) {
    float scale = (max - min) / 255.0;
    for (int i = 0; i < size; i++) {
        output[i] = ((input[i] + 128) * scale) + min;
    }
}

// ==================================================================================== //
//                                    log quantize
// ==================================================================================== //


// 对数量化 (int16_t)
void log_quantize_float32_to_int16(float *input, int16_t *output, int size) {
    for (int i = 0; i < size; i++) {
        // 量化使用对数
        output[i] = (int16_t)(roundf(logf(input[i] + 1e-9) * 1000)); // 缩放值使其符合 int16_t 范围
    }
}

// 对数量化反量化 (int16_t)
void log_dequantize_int16_to_float32(int16_t *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        // 反量化使用指数函数
        output[i] = expf(input[i] / 1000.0);
    }
}

// 对数量化 (int8_t)
void log_quantize_float32_to_int8(float *input, int8_t *output, int size) {
    for (int i = 0; i < size; i++) {
        // 量化使用对数
        output[i] = (int8_t)(roundf(logf(input[i] + 1e-9) * 100)); // 缩放值使其符合 int8_t 范围
    }
}

// 对数量化反量化 (int8_t)
void log_dequantize_int8_to_float32(int8_t *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        // 反量化使用指数函数
        output[i] = expf(input[i] / 100.0);
    }
}