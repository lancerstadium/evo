// ==================================================================================== //
//                                 evo/dev/cpu/kernel.h
// ==================================================================================== //

#ifndef __EVO_DEV_CPU_KERNEL_H__
#define __EVO_DEV_CPU_KERNEL_H__

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                cpu kernel: bare machine
// ==================================================================================== //

// ==================================================================================== //
//                                optim
// ==================================================================================== //

void SGD_update_float32_cpu(float *weights, float *grads, int size, float learning_rate);

// ==================================================================================== //
//                                blas
// ==================================================================================== //

void Gemm_forward_float32_cpu(float *A, float *B, float *C, float *Y, float alpha, float beta, unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type);
void Conv2d_forward_float32_cpu(float *X, float *K, float *Y, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W);
void Conv2d_backward_float32_cpu(float *X, float *K, float *dY, float *dX, float *dK, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W);
void Deconv2d_forward_float32_cpu(float *X, float *K, float *Y, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W);
void Deconv2d_backward_float32_cpu(float *X, float *K, float *dY, float *dX, float *dK, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W);
void Resize_nearest_uint8_cpu(uint8_t *X, uint8_t *Y, int N, int C, int H, int W, int stride, float scale, bool is_forward);
void Resize_nearest_float32_cpu(float *X, float *Y, int N, int C, int H, int W, int stride, float scale, bool is_forward);
void Resize_linear_float32_cpu(float *X, float* Y, int N, int C, int in_H, int in_W, int out_H, int out_W);

// ==================================================================================== //
//                                activate
// ==================================================================================== //

void PRelu_forward_int32_cpu(int32_t *A, int32_t *B, int32_t *Y, unsigned N);
void PRelu_forward_int64_cpu(int64_t *A, int64_t *B, int64_t *Y, unsigned N);
void PRelu_forward_uint32_cpu(uint32_t *A, uint32_t *B, uint32_t *Y, unsigned N);
void PRelu_forward_uint64_cpu(uint64_t *A, uint64_t *B, uint64_t *Y, unsigned N);
void PRelu_forward_float32_cpu(float *A, float *B, float *Y, unsigned N);
void PRelu_forward_float64_cpu(double *A, double*B, double *Y, unsigned N);
void PRelu_backward_float32_cpu(float *A, float *B, float *dY, float *dA, float *dB, unsigned N);
void PRelu_backward_float64_cpu(double *A, double *B, double *dY, double *dA, double *dB, unsigned N);


// ==================================================================================== //
//                                quant & dequant
// ==================================================================================== //

void Quant_symmetric_float32_to_int16_cpu(float *input, int16_t *output, int size, float scale);
void Dequant_symmetric_int16_to_float32_cpu(int16_t *input, float *output, int size, float scale);
void Quant_symmetric_float32_to_int8_cpu(float *input, int8_t *output, int size, float scale);
void Dequant_symmetric_int8_to_float32_cpu(int8_t *input, float *output, int size, float scale);
void Quant_symmetric_float32_to_uint8_cpu(float *input, uint8_t *output, int size, float scale);
void Dequant_symmetric_uint8_to_float32_cpu(uint8_t *input, float *output, int size, float scale);
void Quant_asymmetric_float32_to_int16_cpu(float *input, int16_t *output, int size, float scale, int16_t zero_point);
void Dequant_asymmetric_int16_to_float32_cpu(int16_t *input, float *output, int size, float scale, int16_t zero_point);
void Quant_asymmetric_float32_to_int8_cpu(float *input, int8_t *output, int size, float scale, int8_t zero_point);
void Dequant_asymmetric_int8_to_float32_cpu(int8_t *input, float *output, int size, float scale, int8_t zero_point);
void Quant_asymmetric_float32_to_uint8_cpu(float *input, uint8_t *output, int size, float scale, uint8_t zero_point);
void Dequant_asymmetric_uint8_to_float32_cpu(uint8_t *input, float *output, int size, float scale, uint8_t zero_point);
void Quant_asymmetric_float32_to_int32_cpu(float *input, int32_t *output, int size, float scale, int32_t zero_point);
void Quant_dynrange_float32_to_int16_cpu(float *input, int16_t *output, int size);
void Dequant_dynrange_int16_to_float32_cpu(int16_t *input, float *output, int size, float min, float max);
void Quant_dynrange_float32_to_int8_cpu(float *input, int8_t *output, int size);
void Dequant_dynrange_int8_to_float32_cpu(int8_t *input, float *output, int size, float min, float max);
void Quant_log_float32_to_int16_cpu(float *input, int16_t *output, int size);
void Dequant_log_int16_to_float32_cpu(int16_t *input, float *output, int size);
void Quant_log_float32_to_int8_cpu(float *input, int8_t *output, int size);
void Dequant_log_int8_to_float32_cpu(int8_t *input, float *output, int size);


void Quant_asymmetric_float32_to_int_cpu(float *input, void *output, int size, float scale, int32_t zero_point, int width, int sign);
void Dequant_asymmetric_int_to_float32_cpu(void *input, float *output, int size, float scale, int32_t zero_point, int width, int sign);
void Quant_solve(float* data, int size, float* scale, int* zero_point, int width, int sign, int method);


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CPU_KERNEL_H__
