
// ==================================================================================== //
//                                 evo/dev/cpu/def.h
// ==================================================================================== //

#ifndef __EVO_DEV_CPU_DEF_H__
#define __EVO_DEV_CPU_DEF_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       include
// ==================================================================================== //

#include <evo.h>


// ==================================================================================== //
//                                       cpu device
// ==================================================================================== //

device_t* device_reg_cpu();

// ==================================================================================== //
//                                       cpu kernel
// ==================================================================================== //


void Gemm_forward_float32_cpu(float *A, float *B, float *C, float *Y, float alpha, float beta, unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type);
void Resize_nearest_uint8_cpu(uint8_t *X, uint8_t* Y, int N, int C, int H, int W, int stride, float scale, bool is_forward);
void Resize_nearest_float32_cpu(float *X, float* Y, int N, int C, int H, int W, int stride, float scale, bool is_forward);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CPU_DEF_H__
