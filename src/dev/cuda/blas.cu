#include "cuda_runtime.h"
#include "def.h"
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void sumArraysCPU(float *a, float *b, float *res, const int N) {
    for(int i = 0; i < N; i+=4) {
        res[i] = a[i] + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

__global__ void sumArraysGPU(float *a, float *b, float *res, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        res[i] = a[i] + b[i];
}

extern "C" void arr_sum_cuda() {

    //
    int nElem=1<<24;
    printf("Vector size:%d\n",nElem);
    int nByte=sizeof(float)*nElem;
    float *a_h=(float*)malloc(nByte);
    float *b_h=(float*)malloc(nByte);
    float *res_h=(float*)malloc(nByte);
    float *res_from_gpu_h=(float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    // init data
    for(int i = 0; i < nElem; i++) {
        a_h[i] = 3;
        b_h[i] = 4;
    }

    float *a_d,*b_d,*res_d;
    cudaMalloc((float**)&a_d,nByte);
    cudaMalloc((float**)&b_d,nByte);
    cudaMalloc((float**)&res_d,nByte);

    // copy data
    cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice);


    dim3 block(512);
    dim3 grid((nElem-1)/block.x+1);

    // timer
    double iStart, iElaps;
    iStart = cpuSecond();
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(res_h, res_d, nElem, cudaMemcpyDeviceToHost);

    printf("timer gpu total: %f \n", iElaps);

    iStart = cpuSecond();
    sumArraysCPU(a_h, b_h, res_h, nElem);
    iElaps = cpuSecond() - iStart;

    printf("timer cpu total: %f \n", iElaps);
    
    for(int i = 0; i < 10; i++) {
        printf("%f ", res_h[i]);
    }

    // free
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
}