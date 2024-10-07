#include <evo/dev/cpu/def.h>
#include <math.h>

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

void Resize_nearest_uint8_cpu(uint8_t *X, uint8_t* Y, int N, int C, int H, int W, float scale, bool is_forward) {
    int i, j, k, b;
    int new_h = H * scale;
    int new_w = W * scale;

    for (b = 0; b < N; ++b) {
        for (k = 0; k < C; ++k) {
            for (j = 0; j < new_h; ++j) {
                for (i = 0; i < new_w; ++i) {
                    int in_index = b * H * W * C + k * H * W + (j / scale) * W + (i / scale);
                    int out_index = b * new_h * new_w * C + k * new_h * new_w + j * new_w + i;
                    if (is_forward) {
                        // 上采样
                        Y[out_index] = X[in_index];
                    } else {
                        // 下采样
                        X[in_index] += Y[out_index];
                    }
                }
            }
        }
    }
}

void Resize_nearest_float32_cpu(float *X, float* Y, int N, int C, int H, int W, float scale, bool is_forward) {
    int i, j, k, b;
    int new_h = H * scale;
    int new_w = W * scale;

    for (b = 0; b < N; ++b) {
        for (k = 0; k < C; ++k) {
            for (j = 0; j < new_h; ++j) {
                for (i = 0; i < new_w; ++i) {
                    int in_index = b * H * W * C + k * H * W + (j / scale) * W + (i / scale);
                    int out_index = b * new_h * new_w * C + k * new_h * new_w + j * new_w + i;
                    if (is_forward) {
                        // 上采样
                        Y[out_index] = X[in_index];
                    } else {
                        // 下采样
                        X[in_index] += Y[out_index];
                    }
                }
            }
        }
    }
}