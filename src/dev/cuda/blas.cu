#include <evo/dev/cuda/def.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>

/**
 * kernels:
 *  - [ ] Add           : f <float32> | b <float32>
 *  - [ ] Sum           : f <float32> | b <float32>
 *  - [ ] Scale         : f <float32> | b <float32>
 * 
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
//                                       cuda blas kernels
// ==================================================================================== //


__global__ void cudaSumFloat32(float *a, float *b, float *res, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        res[i] = a[i] + b[i];
}


// ==================================================================================== //
//                                       cuda blas API
// ==================================================================================== //


extern "C" void cuda_sum_float32(float *a_h, float *b_h, float *res_h, int nElem) {
    int nByte = sizeof(float) * nElem;

    float *a_d, *b_d, *res_d;
    cudaMalloc((float **)&a_d, nByte);
    cudaMalloc((float **)&b_d, nByte);
    cudaMalloc((float **)&res_d, nByte);

    cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice);

    dim3 block(512);
    dim3 grid((nElem - 1) / block.x + 1);

    cudaSumFloat32<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaDeviceSynchronize();
    cudaMemcpy(res_h, res_d, nElem, cudaMemcpyDeviceToHost);

    // free
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);
}