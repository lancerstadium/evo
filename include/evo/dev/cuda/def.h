// ==================================================================================== //
//                                 evo/dev/cuda/def.h
// ==================================================================================== //

#ifndef __EVO_DEV_CUDA_DEF_H__
#define __EVO_DEV_CUDA_DEF_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       include
// ==================================================================================== //

#include <evo.h>

// ==================================================================================== //
//                                       cuda blas
// ==================================================================================== //

#ifdef EVO_ACC_CUDA

#define EVO_CUDA_BLOCK 512

// ==================================================================================== //
//                                       cuda device
// ==================================================================================== //

device_t* device_reg_cuda();

// ==================================================================================== //
//                                       cuda kernel
// ==================================================================================== //

void Sum_forward_float32_cuda(float *a, float *b, float *c, int nElem);
void Gemm_forward_float32_cuda(float* A, float* B, float* C, float* Y, float alpha, float beta, unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type);
void Relu_forward_float32_cuda(float *x, int n);

#endif // EVO_ACC_CUDA

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CUDA_DEF_H__
