#include <evo/dev/cuda/def.h>
#include <evo/util/sys.h>
#include <evo/util/log.h>



// ==================================================================================== //
//                                       cuda: allocator
// ==================================================================================== //

void* cuda_alloc(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void cuda_release(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memory free failed: %s\n", cudaGetErrorString(err));
    }
}

// ==================================================================================== //
//                                       cuda: define
// ==================================================================================== //

static allocator_t cuda_alc = {
    .alloc      = cuda_alloc,
    .release    = cuda_release
};

static device_t cuda_dev = {
    .name = "cuda",
    .itf  = NULL,
    .alc  = &cuda_alc,
    .scd  = NULL
};

// ==================================================================================== //
//                                       cuda: API
// ==================================================================================== //

device_t* device_reg_cuda() {
    device_reg_dev(&cuda_dev);
    return internal_device_find("cuda");
}