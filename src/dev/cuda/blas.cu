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
                                       unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type) {
    unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;  // 行索引
    unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;  // 列索引

    if (m >= M || n >= N)
        return;

    float sum = 0;
    for (unsigned int k = 0; k < K; ++k) {
        // 根据 transA 和 transB 处理转置逻辑
        float a_val = transA ? A[k * M + m] : A[m * K + k];  // A 是否转置
        float b_val = transB ? B[n * K + k] : B[k * N + n];  // B 是否转置

        sum += a_val * b_val;
    }

    // 计算乘积结果并应用 alpha
    float res = alpha * sum;

    // 根据 broadcast_type 处理 C 的广播
    if (C != NULL && beta != 0) {
        if (broadcast_type == 1) {  // 标量广播
            res += beta * C[0];
        } else if (broadcast_type == 2) {  // 行向量广播
            res += beta * C[n];
        } else if (broadcast_type == 3) {  // 列向量广播
            res += beta * C[m];
        } else if (broadcast_type == 4) {  // 完整矩阵，无需广播
            res += beta * C[m * N + n];
        }
    }

    // 将结果写入输出矩阵 Y
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
                                       unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type) {


    float *A_d, *B_d, *C_d = NULL, *Y_d;

    // 分配 A 和 B 的内存
    cudaMalloc((void**)&A_d, M * K * sizeof(float));
    cudaMalloc((void**)&B_d, K * N * sizeof(float));
    
    // 分配 Y 的内存
    cudaMalloc((void**)&Y_d, M * N * sizeof(float));

    // 将 A 和 B 复制到设备
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 根据 broadcast_type 分配和复制 C，如果 C 不为 NULL
    if (C != NULL) {
        if (broadcast_type == 1) {  // 标量
            cudaMalloc((void**)&C_d, sizeof(float));
            cudaMemcpy(C_d, C, sizeof(float), cudaMemcpyHostToDevice);
        } else if (broadcast_type == 2) {  // 行向量
            cudaMalloc((void**)&C_d, N * sizeof(float));
            cudaMemcpy(C_d, C, N * sizeof(float), cudaMemcpyHostToDevice);
        } else if (broadcast_type == 3) {  // 列向量
            cudaMalloc((void**)&C_d, M * sizeof(float));
            cudaMemcpy(C_d, C, M * sizeof(float), cudaMemcpyHostToDevice);
        } else if (broadcast_type == 4) {  // 完整矩阵
            cudaMalloc((void**)&C_d, M * N * sizeof(float));
            cudaMemcpy(C_d, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    // 设置 CUDA 网格和块的大小
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // 启动 CUDA 核函数
    cudaGemmForwardFloat32<<<grid, block>>>(A_d, B_d, C_d, Y_d, alpha, beta, M, N, K, transA, transB, broadcast_type);
    cudaDeviceSynchronize();
    cuda_check(cudaGetLastError());

    // 将结果从设备复制到主机
    cudaMemcpy(Y, Y_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(A_d);
    cudaFree(B_d);
    if (C_d != NULL) cudaFree(C_d);
    cudaFree(Y_d);
}