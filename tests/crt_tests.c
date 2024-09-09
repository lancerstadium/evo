#include "sob.h"
#include <evo.h>
#include <math.h>

/**
 * ask hear: https://chatgpt.com/c/94ab7adb-e89e-4341-ba8d-fd6ea4377eaf
 * 
 */

float goodness(tensor_t* h) {
    // 计算每层激活值的平方均值
    float sum = 0.0;
    float* data = h->datas;
    for (int i = 0; i < h->ndata; i++) {
        sum += data[i] * data[i];
    }
    return sum / h->ndata;
}

/**
 * @brief
 * hinton loss:
 * 
 * ```latex
 * 
 *      L(x) = log(1 + e^{y (\theta - G_x)})
 * 
 *  - x         : input image
 *  - y         : class of input x
 *  - \theta    : threshold
 * 
 * ```
 */
float hinton_loss(tensor_t* h_pos, tensor_t* h_neg, float theta, float alpha) {
    float g_pos = goodness(h_pos);
    float g_neg = goodness(h_neg);
    
    // 计算正样本和负样本的损失
    float loss_pos = logf(1 + expf(theta - g_pos)) * alpha;
    float loss_neg = logf(1 + expf(g_neg - theta)) * alpha;
    
    // 返回总的损失
    return loss_pos + loss_neg;
}


/**
 * @brief
 * symba loss:
 * 
 * ```latex
 * 
 *      L(P, N) = log(1 + e^{-\alpha (G_P - G_N)})
 * 
 * ```
 */
float symba_loss(tensor_t* h_pos, tensor_t* h_neg, float alpha) {
    float g_pos = goodness(h_pos);
    float g_neg = goodness(h_neg);
    
    float Delta = g_pos - g_neg;
    return logf(1 + expf(-alpha * Delta));
}

/**
 *  ** Mnist Model ** :
 *      - Flatten           : [1,1,28,28] ->     [1,784]
 *      - FC                :     [1,784] ->     [1,500]
 *      - ReLU              :     [1,500] ->     [1,500]
 *      - FC                :     [1,500] ->      [1,10]
 *      - Softmax           :      [1,10] ->      [1,10]
 *
 *  ** Train Config ** :
 *      - learning_rate     = 0.1(hinton_loss) else 0.35
 *      - optimizer         = Adam
 *      - num_epochs        = 600(hinton_loss) else 60
 *      - num_batchs        = 4096
 */
model_t* mnist_model() {
    model_t* mdl = model_new("mnist_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28});
    // graph_add_conv2d(mdl->graph, (int64_t[]){3, 3}, NULL, NULL, NULL, 0, NULL);
    // graph_add_resize(mdl->graph, (float[]){1, 1, 0.5, 0.5}, 4, "bilinear");
    graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, NULL, NULL, NULL, 0, 0);
    graph_add_flatten(mdl->graph);
    graph_add_linear(mdl->graph, 500, "relu");
    graph_add_linear(mdl->graph, 10, "softmax");
    return mdl;
}

UnitTest_fn_def(test_model_create) {
    device_reg("cpu");
    // Dataset
    const char* image_filename = "picture/mnist/t10k-images-idx3-ubyte";
    const char* label_filename = "picture/mnist/t10k-labels-idx1-ubyte";
    image_t* imgs = image_load_mnist(image_filename, label_filename);
    if(!imgs) {
        fprintf(stderr, "Load mnist fail, please exec `download_mnist.sh` in Dir `picture`.\n");
        return "Load Mnist Fail!";
    } else {
        fprintf(stderr, "Load Mnist Success!\n");
    }
    attribute_t* label = image_get_attr(imgs, "label");

    // Model
    model_t* mdl = mnist_model();
    graph_dump2(mdl->graph);
    model_show_tensors(mdl);

    // Train
    tensor_t *x_tmp, *x;
    
    int num_epochs = 8;
    int num_batchs = 20;
    int learning_rate = 0.1;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Mini-batch training
        for (int b = 0; b < num_batchs; b++) {
            x_tmp = image_get_raw(imgs, b);
            x  = tensor_cast(x_tmp, TENSOR_TYPE_FLOAT32);
            uint8_t y = label->bs[b];
            model_set_tensor(mdl, "Input0", x);
            model_train_label(mdl, y);
        }

        // Evaluate the model on the training and test set
        if (epoch % 2 == 0) {
            float train_error = 1.0;
            float test_error = 1.0;
            int acc_cnt = 0;
            for(int b = 0; b < num_batchs; b++) {
                x_tmp = image_get_raw(imgs, b);
                x  = tensor_cast(x_tmp, TENSOR_TYPE_FLOAT32);
                // image_dump_raw(imgs, b);
                // fprintf(stderr, "%u\n", label->bs[b]);
                uint8_t y = label->bs[b];
                tensor_t* y_ts = model_eval(mdl, x);
                tensor_t* y_out = tensor_argmax(y_ts, 0, 1, 0);
                acc_cnt += (((float*)y_out->datas)[0] == (float)y) ? 1 : 0;
            }
            // tensor_t* sss = model_get_tensor(mdl, "Gemm3_out0");
            // tensor_dump2(sss->grad);
            train_error -= (acc_cnt / num_batchs);
            printf("[%4d] Train acc: %.2f%%, Test acc: %.2f%%\n", epoch, train_error * 100, test_error * 100);
        }
    }
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);