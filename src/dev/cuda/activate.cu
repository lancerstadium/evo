#include <evo/dev/cuda/def.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>

/**
 * kernels:
 *  - [*] Fill          : <float32>
 *  - [*] Relu          : <float32>
 *  - [*] LeakyRelu     : <float32>
 * 
 * ref: 
 *  1. https://blog.csdn.net/hehe199807/article/details/108365427
 *  2. https://zhuanlan.zhihu.com/p/688627970
 */

// ==================================================================================== //
//                                       cuda tool
// ==================================================================================== //

dim3 cuda_grid(size_t n) {
    size_t k = (n - 1) / EVO_CUDA_BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * EVO_CUDA_BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    // printf("%ld %ld %ld %ld\n", n, x, y, x*y*EVO_CUDA_BLOCK);
    return d;
}

void cuda_check(cudaError_t status) {
    // cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        perror(buffer);
        assert(0);
        exit(-1);
    }
    if (status2 != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        perror(buffer);
        assert(0);
        exit(-1);
    }
}


// ==================================================================================== //
//                                       cuda activate kernels
// ==================================================================================== //

__device__ float reluForwardKernelFloat32(float x) {
    return x*(x>0);
}

__device__ float leakyReluForwardKernelFloat32(float x) {
    return (x>0) ? x : 0.1f*x;
}

__global__ void reluForwardFloat32(float *x, int n) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = reluForwardKernelFloat32(x[i]);
}


// ==================================================================================== //
//                                       cuda activate API
// ==================================================================================== //

extern "C" void Relu_forward_float32_cuda(float *x, int n) {
    reluForwardFloat32<<<cuda_grid(n), EVO_CUDA_BLOCK>>>(x, n);
}