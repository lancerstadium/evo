#include <evo/dev/cpu/kernel.h>


// ==================================================================================== //
//                                       activate: PRelu
// ==================================================================================== //

void PRelu_forward_int32_cpu(int32_t *A, int32_t *B, int32_t *Y, unsigned N) {
    for(unsigned i = 0; i < N; i++) {
        Y[i] = (A[i] < 0) ? A[i] * B[i] : A[i];
    }
}

void PRelu_forward_int64_cpu(int64_t *A, int64_t *B, int64_t *Y, unsigned N) {
    for(unsigned i = 0; i < N; i++) {
        Y[i] = (A[i] < 0) ? A[i] * B[i] : A[i];
    }
}

void PRelu_forward_uint32_cpu(uint32_t *A, uint32_t *B, uint32_t *Y, unsigned N) {
    for(unsigned i = 0; i < N; i++) {
        Y[i] = (A[i] < 0) ? A[i] * B[i] : A[i];
    }
}

void PRelu_forward_uint64_cpu(uint64_t *A, uint64_t *B, uint64_t *Y, unsigned N) {
    for(unsigned i = 0; i < N; i++) {
        Y[i] = (A[i] < 0) ? A[i] * B[i] : A[i];
    }
}

void PRelu_forward_float32_cpu(float *A, float *B, float *Y, unsigned N) {
    for(unsigned i = 0; i < N; i++) {
        Y[i] = (A[i] < 0) ? A[i] * B[i] : A[i];
    }
}

void PRelu_forward_float64_cpu(double *A, double*B, double *Y, unsigned N) {
    for(unsigned i = 0; i < N; i++) {
        Y[i] = (A[i] < 0) ? A[i] * B[i] : A[i];
    }
}