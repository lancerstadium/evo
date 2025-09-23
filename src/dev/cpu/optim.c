#include <evo/dev/cpu/kernel.h>




void SGD_update_float32_cpu(float *weights, float *grads, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * grads[i];
    }
}