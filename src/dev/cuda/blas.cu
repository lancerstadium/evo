#include <evo/dev/cuda/def.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>

/**
 * kernels:
 *  - [ ] Add           : f <float32> | b <float32>
 *  - [ ] Sum           : f <float32> | b <float32>
 *  - [ ] Scale         : f <float32> | b <float32>
 *  - [ ] Gemm          : f <float32> | b <float32>
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


__global__ void cudaSumForwardFloat32(float *a, float *b, float *res, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        res[i] = a[i] + b[i];
}


__global__ void cudaGemmForwardFloat32(float* A, float* B, float* C, float* Y, float alpha, float beta, 
                                       unsigned M, unsigned N, unsigned K, int transA, int transB) {
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;  // Row index of Y (or A)
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;  // Column index of Y (or B)
    
    if (m >= M || n >= N)
        return;

    float c = 0;
    for (unsigned k = 0; k < K; ++k) {
        // Handle different cases of transA and transB
        float a_val = transA ? A[k * M + m] : A[m * K + k]; // Transpose A if transA is true
        float b_val = transB ? B[n * K + k] : B[k * N + n]; // Transpose B if transB is true
        
        c += a_val * b_val;
    }

    // Apply alpha and calculate final result
    float res = alpha * c;
    
    // Apply beta and add bias if C is not NULL
    if (C != NULL && beta != 0) {
        res += beta * C[m * N + n];
    }

    // Write result to output matrix Y
    Y[m * N + n] = res;
}


// ==================================================================================== //
//                                       cuda blas API
// ==================================================================================== //


extern "C" void Sum_forward_float32_cuda(float *a, float *b, float *c, int nElem) {
    int nByte = sizeof(float) * nElem;

    float *a_d, *b_d, *c_d;
    cudaMalloc((float **)&a_d, nByte);
    cudaMalloc((float **)&b_d, nByte);
    cudaMalloc((float **)&c_d, nByte);

    cudaMemcpy(a_d, a, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, nByte, cudaMemcpyHostToDevice);

    dim3 block(512);
    dim3 grid((nElem - 1) / block.x + 1);

    cudaSumForwardFloat32<<<grid, block>>>(a_d, b_d, c_d, nElem);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_d, nElem, cudaMemcpyDeviceToHost);

    // free
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


extern "C" void Gemm_forward_float32_cuda(float* A, float* B, float* C, float* Y, float alpha, float beta, 
                                       unsigned M, unsigned N, unsigned K, int transA, int transB) {

    float *A_d, *B_d, *C_d, *Y_d;
    cudaMalloc((void**)&A_d, M * K * sizeof(float));
    cudaMalloc((void**)&B_d, K * N * sizeof(float));
    cudaMalloc((void**)&C_d, M * N * sizeof(float));
    cudaMalloc((void**)&Y_d, M * N * sizeof(float));
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y_d, C, M * N * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(32, 32);
    dim3 grid((M - 1)/ block.x + 1, (N - 1) / block.y + 1);

    cudaGemmForwardFloat32<<<grid, block>>>(A_d, B_d, C_d, Y_d, alpha, beta, M, N, K, transA, transB);
    cudaDeviceSynchronize();
    cudaMemcpy(Y, Y_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // free
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(Y_d);
}