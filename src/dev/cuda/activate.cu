#include <evo/dev/cuda/def.h>
#include <cuda_runtime.h>

dim3 cuda_gridsize(size_t n) {
    size_t k = (n - 1) / EVO_CUDA_BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * EVO_CUDA_BLOCK) + 1;
    }
    dim3 d(x, y);
    // printf("%ld %ld %ld %ld\n", n, x, y, x*y*EVO_CUDA_BLOCK);
    return d;
}

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

extern "C" void cuda_relu_forward_float32(float *x, int n) {
    reluForwardFloat32<<<cuda_gridsize(n), EVO_CUDA_BLOCK>>>(x, n);
}