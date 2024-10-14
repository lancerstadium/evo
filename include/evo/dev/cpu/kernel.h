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
//                                blas
// ==================================================================================== //

void Gemm_forward_float32_cpu(float *A, float *B, float *C, float *Y, float alpha, float beta, unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type);
void Resize_nearest_uint8_cpu(uint8_t *X, uint8_t* Y, int N, int C, int H, int W, int stride, float scale, bool is_forward);
void Resize_nearest_float32_cpu(float *X, float* Y, int N, int C, int H, int W, int stride, float scale, bool is_forward);


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


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CPU_KERNEL_H__
