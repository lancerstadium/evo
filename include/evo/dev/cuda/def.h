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

void cuda_sum_float32(float *a_h, float *b_h, float *res_h, int nElem);
void cuda_relu_forward_float32(float *x, int n);

#endif 

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_DEV_CUDA_DEF_H__
