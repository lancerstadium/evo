#include <evo/dev/cpu/kernel.h>


// ==================================================================================== //
//                                       blas: Gemm
// ==================================================================================== //

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


void Conv2d_forward_float32_cpu(float *X, float *K, float *Y, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W) {
    for (int i = 0; i < Y_H; i++) {
        for (int j = 0; j < Y_W; j++) {
            Y[i * Y_W + j] = 0; // 初始化输出
            for (int m = 0; m < K_H; m++) {
                for (int n = 0; n < K_W; n++) {
                    Y[i * Y_W + j] += X[(i + m) * X_W + (j + n)] * K[m * K_W + n];
                }
            }
        }
    }
}

void Conv2d_backward_float32_cpu(float *X, float *K, float *dY, float *dX, float *dK, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W) {
    // 计算 dX 的梯度
    for (int i = 0; i < X_H; i++) {
        for (int j = 0; j < X_W; j++) {
            dX[i * X_W + j] = 0; // 初始化 dX
        }
    }

    for (int i = 0; i < Y_H; i++) {
        for (int j = 0; j < Y_W; j++) {
            for (int m = 0; m < K_H; m++) {
                for (int n = 0; n < K_W; n++) {
                    dX[(i + m) * X_W + (j + n)] += dY[i * Y_W + j] * K[m * K_W + n];
                }
            }
        }
    }

    // 计算 dK 的梯度
    for (int m = 0; m < K_H; m++) {
        for (int n = 0; n < K_W; n++) {
            dK[m * K_W + n] = 0; // 初始化 dK
            for (int i = 0; i < Y_H; i++) {
                for (int j = 0; j < Y_W; j++) {
                    dK[m * K_W + n] += dY[i * Y_W + j] * X[(i + m) * X_W + (j + n)];
                }
            }
        }
    }
}

void Deconv2d_forward_float32_cpu(float *X, float *K, float *Y, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W) {
    for (int i = 0; i < Y_H; i++) {
        for (int j = 0; j < Y_W; j++) {
            Y[i * Y_W + j] = 0; // 初始化输出
        }
    }

    for (int i = 0; i < X_H; i++) {
        for (int j = 0; j < X_W; j++) {
            for (int m = 0; m < K_H; m++) {
                for (int n = 0; n < K_W; n++) {
                    Y[(i + m) * Y_W + (j + n)] += X[i * X_W + j] * K[m * K_W + n];
                }
            }
        }
    }
}


void Deconv2d_backward_float32_cpu(float *X, float *K, float *dY, float *dX, float *dK, int X_H, int X_W, int K_H, int K_W, int Y_H, int Y_W) {
    // 初始化梯度
    for (int i = 0; i < X_H; i++) {
        for (int j = 0; j < X_W; j++) {
            dX[i * X_W + j] = 0; // 初始化 dX
        }
    }

    for (int m = 0; m < K_H; m++) {
        for (int n = 0; n < K_W; n++) {
            dK[m * K_W + n] = 0; // 初始化 dK
        }
    }

    // 反卷积的反向传播
    for (int i = 0; i < Y_H; i++) {
        for (int j = 0; j < Y_W; j++) {
            for (int m = 0; m < K_H; m++) {
                for (int n = 0; n < K_W; n++) {
                    if ((i - m) >= 0 && (j - n) >= 0 && (i - m) < X_H && (j - n) < X_W) {
                        dX[(i - m) * X_W + (j - n)] += dY[i * Y_W + j] * K[m * K_W + n];
                        dK[m * K_W + n] += dY[i * Y_W + j] * X[(i - m) * X_W + (j - n)];
                    }
                }
            }
        }
    }
}





// ==================================================================================== //
//                                       blas: Resize
// ==================================================================================== //

void Resize_nearest_uint8_cpu(uint8_t *X, uint8_t* Y, int N, int C, int H, int W, int stride, float scale, bool is_forward) {
    int new_H = is_forward ? (int)(H * scale) : (int)(H / scale);
    int new_W = is_forward ? (int)(W * scale) : (int)(W / scale);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < new_H; ++i) {
                for (int j = 0; j < new_W; ++j) {
                    int in_i = is_forward ? (int)(i / scale) : (int)(i * scale);
                    int in_j = is_forward ? (int)(j / scale) : (int)(j * scale);
                    
                    // 确保索引在原始图像的范围内
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
                        int index = n * C * H * W + c * H * W + in_i * W + in_j;
                        int new_index = n * C * new_H * new_W + c * new_H * new_W + i * new_W + j;
                        Y[new_index] = X[index];
                    } else {
                        // 如果索引超出原始图像范围，则可以根据需要设置默认值，这里设置为0
                        int new_index = n * C * new_H * new_W + c * new_H * new_W + i * new_W + j;
                        Y[new_index] = 0.0f;
                    }
                }
            }
        }
    }
}

void Resize_nearest_float32_cpu(float *X, float* Y, int N, int C, int H, int W, int stride, float scale, bool is_forward) {
    int new_H = is_forward ? (int)(H * scale) : (int)(H / scale);
    int new_W = is_forward ? (int)(W * scale) : (int)(W / scale);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < new_H; ++i) {
                for (int j = 0; j < new_W; ++j) {
                    int in_i = is_forward ? (int)(i / scale) : (int)(i * scale);
                    int in_j = is_forward ? (int)(j / scale) : (int)(j * scale);
                    
                    // 确保索引在原始图像的范围内
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
                        int index = n * C * H * W + c * H * W + in_i * W + in_j;
                        int new_index = n * C * new_H * new_W + c * new_H * new_W + i * new_W + j;
                        Y[new_index] = X[index];
                    } else {
                        // 如果索引超出原始图像范围，则可以根据需要设置默认值，这里设置为0
                        int new_index = n * C * new_H * new_W + c * new_H * new_W + i * new_W + j;
                        Y[new_index] = 0.0f;
                    }
                }
            }
        }
    }
}