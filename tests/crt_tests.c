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

float hinton_loss(tensor_t* h_pos, tensor_t* h_neg, float theta, float alpha) {
    float g_pos = goodness(h_pos);
    float g_neg = goodness(h_neg);
    
    // 计算正样本和负样本的损失
    float loss_pos = logf(1 + expf(theta - g_pos)) * alpha;
    float loss_neg = logf(1 + expf(g_neg - theta)) * alpha;
    
    // 返回总的损失
    return loss_pos + loss_neg;
}

float symba_loss(tensor_t* h_pos, tensor_t* h_neg, float alpha) {
    float g_pos = goodness(h_pos);
    float g_neg = goodness(h_neg);
    
    float Delta = g_pos - g_neg;
    return logf(1 + expf(-alpha * Delta));
}

void update_parameters(node_t* layer, float loss, float learning_rate) {
    // 获取权重和偏置的tensor
    tensor_t* weights = layer->in[1];
    tensor_t* bias = layer->in[2];
    float* w_d = weights->datas;
    float* b_d = bias->datas;
    
    // 假设权重和偏置都是1D数组，按元素更新它们
    for (int i = 0; i < weights->ndata; i++) {
        w_d[i] -= learning_rate * loss;  // 使用损失直接调整权重
    }
    for (int i = 0; i < bias->ndata; i++) {
        b_d[i] -= learning_rate * loss;  // 使用损失直接调整偏置
    }
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
    graph_add_resize(mdl->graph, (float[]){1, 1, 0.5, 0.5}, 4, "bilinear");
    graph_add_flatten(mdl->graph);
    graph_add_dense(mdl->graph, 500, "relu");
    graph_add_dense(mdl->graph, 10, "softmax");
    return mdl;
}

UnitTest_fn_def(test_model_create) {
    device_reg("cpu");
    // Dataset
    const char* image_filename = "picture/mnist/t10k-images-idx3-ubyte";
    const char* label_filename = "picture/mnist/t10k-labels-idx1-ubyte";
    image_t* imgs = image_load_mnist(image_filename, label_filename);
    // int idx_train[100], idx_test[20];
    // tensor_t* data_train = image_get_raw_batch(imgs, sizeof(idx_train) / sizeof(int), idx_train);
    // tensor_t* data_test  = image_get_raw_batch(imgs, sizeof(idx_test)  / sizeof(int), idx_test);
    // tensor_t* f32_train  = tensor_cast(data_train, TENSOR_TYPE_FLOAT32);
    // tensor_t* f32_test   = tensor_cast(data_test , TENSOR_TYPE_FLOAT32);

    // Model
    model_t* mdl = mnist_model();
    graph_dump(mdl->graph);
    model_dump_tensor(mdl); 
    
    // Train
    tensor_t *x_tmp, *x;
    attribute_t* label = image_get_attr(imgs, "label");

    int num_epochs = 600;
    int num_batchs = 4096;
    int learning_rate = 0.1;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Mini-batch training
        for (int batch = 0; batch < num_batchs; batch++) {
            x_tmp = image_get_raw(imgs, batch);
            x  = tensor_cast(x_tmp, TENSOR_TYPE_FLOAT32);
            uint8_t y = label->bs[batch];

            // // Positive and negative examples
            // tensor_t* x_pos = make_examples(mdl, x, y, 1);  // 正样本
            // tensor_t* x_neg = make_examples(mdl, x, y, 0);  // 负样本

            // Train layers in turn
            for (int layer_idx = 0; layer_idx < mdl->graph->nnode; layer_idx++) {
                // node_t* layer = mdl->graph->nodes;
                // tensor_t* h_pos = node_fotward(layer, x_pos);
                // tensor_t* h_neg = node_forward(layer, x_neg);

                // // 计算损失
                // float loss = hinton_loss(h_pos, h_neg, 2.0, 1.0);
                
                // 参数更新（不使用反向传播）
                // update_parameters(layer, loss, learning_rate);
                
                // 更新正样本和负样本
                // x_pos = node_forward(layer, x_pos);
                // x_neg = node_forward(layer, x_neg);
            }
        }

        // Evaluate the model on the training and test set
        if (epoch % 5 == 0) {
            float train_error = 0.0;
            float test_error = 0.0;
            printf("[%4d] Training: %.2f%%, Test: %.2f%%\n", epoch, train_error * 100, test_error * 100);
        }
    }

    // Inference
    tensor_t * in = model_get_tensor(mdl, "Input0");
    for(int i = 0; i < 10; i++) {
        x_tmp = image_get_raw(imgs, i);
        x  = tensor_cast(x_tmp, TENSOR_TYPE_FLOAT32);
        uint8_t y = label->bs[i];
        tensor_copy(in, x);
        graph_prerun(mdl->graph);
        graph_run(mdl->graph);
        graph_posrun(mdl->graph);
    }

    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);