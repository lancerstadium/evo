#include <evo/dev/cpu/def.h>

void Gemm_forward_float32_cpu(float *A, float *B, float *C, float *Y, float alpha, float beta,
                              unsigned M, unsigned N, unsigned K, int transA, int transB) {
    unsigned i, j, k;
    float sum;
    int oa, ob, oy;  // 用于矩阵 A, B, Y 的索引

    // 初始化索引
    oa = ob = oy = 0;

    // 处理矩阵 A 和 B 的转置情况
    if (transA && transB) {
        // A 和 B 都转置
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                sum = 0;
                for (k = 0; k < K; k++) {
                    sum += A[k * M + i] * B[j * K + k];
                }
                Y[oy] = alpha * sum;
                // 处理偏置 C
                if (C != NULL && beta != 0) {
                    Y[oy] += beta * C[oy];
                }
                oy++;
            }
        }
    } else if (transA) {
        // 仅 A 转置
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                sum = 0;
                for (k = 0; k < K; k++) {
                    sum += A[k * M + i] * B[k * N + j];
                }
                Y[oy] = alpha * sum;
                // 处理偏置 C
                if (C != NULL && beta != 0) {
                    Y[oy] += beta * C[oy];
                }
                oy++;
            }
        }
    } else if (transB) {
        // 仅 B 转置
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                sum = 0;
                for (k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[j * K + k];
                }
                Y[oy] = alpha * sum;
                // 处理偏置 C
                if (C != NULL && beta != 0) {
                    Y[oy] += beta * C[oy];
                }
                oy++;
            }
        }
    } else {
        // 无转置
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                sum = 0;
                for (k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                Y[oy] = alpha * sum;
                // 处理偏置 C
                if (C != NULL && beta != 0) {
                    Y[oy] += beta * C[oy];
                }
                oy++;
            }
        }
    }
}
