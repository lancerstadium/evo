#include <evo/dev/cpu/def.h>

void Gemm_forward_float32_cpu(float *A, float *B, float *C, float *Y, float alpha, float beta, 
                              unsigned M, unsigned N, unsigned K, int transA, int transB, int broadcast_type) {
    unsigned i, j, k;
    float sum;
    int oy = 0;

    // 根据转置情况计算矩阵元素
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k = 0; k < K; k++) {
                // Handle different cases of transA and transB
                float a_val = transA ? A[k * M + i] : A[i * K + k];  // Transpose A if transA is true
                float b_val = transB ? B[j * K + k] : B[k * N + j];  // Transpose B if transB is true
                sum += a_val * b_val;
            }

            // Apply alpha and calculate the final result for Y
            Y[oy] = alpha * sum;

            // 运行时处理 C 的广播
            if (C != NULL && beta != 0) {
                if (broadcast_type == 1) {
                    Y[oy] += beta * C[0];  // 标量广播
                } else if (broadcast_type == 2) {
                    Y[oy] += beta * C[j];  // 行向量广播
                } else if (broadcast_type == 3) {
                    Y[oy] += beta * C[i];  // 列向量广播
                } else if (broadcast_type == 4) {
                    Y[oy] += beta * C[oy];  // 完整矩阵，无需广播
                }
            }

            oy++;  // Increment output matrix index
        }
    }
}