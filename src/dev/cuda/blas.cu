#include <evo/dev/cuda/def.h>
#include <cuda_runtime.h>


// ref: https://blog.csdn.net/hehe199807/article/details/108365427
// ref: https://zhuanlan.zhihu.com/p/688627970

__global__ void cudaSumFloat32(float *a, float *b, float *res, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        res[i] = a[i] + b[i];
}

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