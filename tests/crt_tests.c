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
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28}, false);
    // graph_add_resize(mdl->graph, (float[]){1, 1, 0.5, 0.5}, 4, "bilinear");
    // graph_add_conv2d(mdl->graph, (int64_t[]){3, 3}, NULL, NULL, NULL, 0, NULL);
    // graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, NULL, NULL, NULL, 0, 0);
    graph_add_flatten(mdl->graph);
    graph_add_linear(mdl->graph, 500, true, "relu");
    graph_add_linear(mdl->graph, 10, true, "softmax");
    return mdl;
}

UnitTest_fn_def(test_mnist_create) {
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
    graph_dump(mdl->graph);
    model_show_tensors(mdl);

    // Train
    trainer_t* trn = trainer_new(1e-6, 1e-8, TRAINER_LOSS_MSE, TRAINER_OPT_SGD);
    tensor_t *x_tmp, *x;
    
    int num_epochs = 20;
    int num_batchs = 800;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Mini-batch training
        for (int b = 0; b < num_batchs; b++) {
            x_tmp = image_get_raw(imgs, b);
            x  = tensor_cast(x_tmp, TENSOR_TYPE_FLOAT32);
            uint8_t y = label->bs[b];
            model_set_tensor(mdl, "Input0", x);
            tensor_t* y_ts = tensor_new_one_hot(2, (int[]){1, 10}, y);
            // image_dump_raw(imgs, b);
            // fprintf(stderr, "<%u> ", y);
            trainer_step(trn, mdl, y_ts);
            // tensor_t* sss = model_get_tensor(mdl, "Gemm1_out0");
            // tensor_dump2(sss);
        }

        // Evaluate the model on the training and test set
        if (epoch % 2 == 0) {
            float train_acc = 0.0;
            float test_acc = 0.0;
            int acc_cnt = 0;
            for(int b = 0; b < num_batchs; b++) {
                x_tmp = image_get_raw(imgs, b);
                x  = tensor_cast(x_tmp, TENSOR_TYPE_FLOAT32);
                // image_dump_raw(imgs, b);
                // fprintf(stderr, "%u\n", label->bs[b]);
                uint8_t y = label->bs[b];
                tensor_t* y_ts = model_eval(mdl, x);
                tensor_t* y_us = tensor_squeeze(y_ts, NULL, 0);
                tensor_t* y_out = tensor_argmax(y_us, 0, 1, 0);
                int64_t yy = ((int64_t*)y_out->datas)[0];
                acc_cnt += ((yy == (int64_t)y) ? 1 : 0);
                // fprintf(stderr, "<%u %ld> ", y, yy);
                // tensor_dump2(y_out);
            }
            train_acc += ((float)acc_cnt / (float)num_batchs);
            fprintf(stderr, "[%4d] Loss: %.2f, Train acc: %.2f%%, Test acc: %.2f%%\n--\n", epoch, trn->cur_loss, train_acc * 100, test_acc * 100);
        }
    }
    return NULL;
}


/**
 * @brief 
 * 
 * ref: https://blog.csdn.net/dbat2015/article/details/48463047
 * ref: https://blog.csdn.net/ECHOSON/article/details/136149724
 * ref: https://www.bilibili.com/video/BV1a14y167vh
 * 
 * ```
 * Input (x1, x2)
 *   ↓
 * Hidden Layer (3 Neurons, ReLU Activation)
 *   ↓
 * Output (1 Neuron, Linear Activation)
 * 
 * ```
 */
model_t* simple_model() {
    model_t* mdl = model_new("simple_model");
    graph_add_input(mdl->graph, 2, (int[]){1, 2}, false);
    node_t* l1 = graph_add_linear(mdl->graph, 3, true, "tanh");
    node_t* l2 = graph_add_linear(mdl->graph, 1, true, NULL);

    // Init Param
    tensor_apply(l1->in[1], (float[]){0.1, 0.2, 0.3, 0.4, 0.5, 0.6}     , 6 * sizeof(float));
    tensor_apply(l1->in[2], (float[]){0.01, 0.02, 0.03}                 , 3 * sizeof(float));
    tensor_apply(l2->in[1], (float[]){0.1, 0.2, 0.3}                    , 3 * sizeof(float));
    tensor_apply(l2->in[2], (float[]){0.05}                             , 1 * sizeof(float));
    return mdl;
}

UnitTest_fn_def(test_simple_create) {
    // Data
    float X[] = {
        -3,     -2,
        -2.7,   -1.8,
        -2.4,   -1.6,
        -2.1,   -1.4,
        -1.8,   -1.2,
        -1.5,   -1,
        -1.2,   -0.8,
        -0.9,   -0.6,
        -0.6,   -0.4,
        -0.3,   -0.2,
           0,   -2.22,
         0.3,    0.2,
         0.6,    0.4,
         0.9,    0.6,
         1.2,    0.8,
         1.5,      1,
         1.8,    1.2,
    };
    float y[] = {
        0.6589,
        0.2206,
        -0.1635,
        -0.4712,
        -0.6858,
        -0.7975,
        -0.804,
        -0.7113,
        -0.5326,
        -0.2875,
        0,
        0.3035,
        0.5966,
        0.8553,
        1.06,
        1.1975,
        1.2618
    };
    int X_off = 2, y_off = 1;
    // 0.205
    
    // Model
    model_t* mdl = simple_model();
    graph_dump(mdl->graph);
    model_show_tensors(mdl);

    // Train
    tensor_t* sss = NULL;
    int nepoch = 10;
    tensor_t* loss_vec = tensor_new("loss", TENSOR_TYPE_FLOAT32);
    tensor_reshape(loss_vec, 2, (int[]){nepoch, 1});
    float* loss_data = loss_vec->datas;
    trainer_t* trn = trainer_new(0.001, 1e-8, TRAINER_LOSS_MSE, TRAINER_OPT_SGD);
    tensor_t* X_ts, *y_ts;
    for(int e = 0; e < nepoch; e++) {
        for(int b = 0; b < sizeof(y)/sizeof(float); b++) {
            X_ts = tensor_new_float32("X", (int[]){1, X_off}, 2, X + b * X_off * 5, X_off * 5);
            y_ts = tensor_new_float32("y", (int[]){1, y_off}, 2, y + b, y_off * 5);
            model_set_tensor(mdl, "Input0", X_ts);
            trainer_step(trn, mdl, y_ts);
            if(e == 0) {
                sss = model_get_tensor(mdl, "Gemm0_out0");
                tensor_dump1(sss);
                sss = model_get_tensor(mdl, "Tanh1_out0");
                tensor_dump1(sss);
                sss = model_get_tensor(mdl, "Gemm2_out0");
                tensor_dump1(sss);
                fprintf(stderr, "--\n");
            }
            trainer_zero_grad(trn, mdl);
            // model_eval(mdl, X_ts);
            // fprintf(stderr, "<%.0f %2.2f> ", y[b], trn->cur_loss);
        }
        // fprintf(stderr, "[%2d] Loss: %.8f\n", e, trn->cur_loss);
        loss_data[e] = trn->cur_loss;
    }

    figure_t* fig = figure_new_1d("loss", FIGURE_TYPE_VECTOR, loss_vec);
    fig->axiss[1]->is_auto_scale = false;
    fig->axiss[1]->range_min = -0.1;
    fig->axiss[1]->range_max = 2;
    figure_save(fig, "loss.svg");

    // Eval
    // for(int b = 0; b < sizeof(y)/sizeof(float); b++) {
    //     X_ts = tensor_new_float32("X", (int[]){1, X_off}, 2, X + b * X_off, X_off);
    //     y_ts = model_eval(mdl, X_ts);
    //     fprintf(stderr, "<%f %f> ", y[b], y_ts->datas ? ((float*)y_ts->datas)[0] : 0.0f);
    //     tensor_t* sss = model_get_tensor(mdl, "Gemm2_out0");
    //     tensor_dump2(sss);
    // }
    return NULL;
}

/**
 * @brief
 * 
 * ref: https://www.bilibili.com/video/BV1QV4y1E7eA
 * 
 * ```
 * Input (x1, x2)
 *   ↓
 * Hidden Layer (2 Neurons, None Activation)
 *   ↓
 * Output (1 Neuron, Linear Activation)
 * 
 * Init:
 *  - [x1, x2]  = [0.5, 1.0]
 *  - [y]       = [0.8]
 *  - l1_w[2,2] = [[1.0, 0.5], [0.5, 0.7]]
 *  - l2_w[2,1] = [[1.0], [2.0]]
 * 
 * Forward 1:
 *  - l1_o[1,2] = [[1, 0.95]]
 *  - l2_o[1,1] = [2.9]
 * 
 * Backward 1:
 *  - loss(MSE) = 2.205
 *  - l1_w[2,2] = [[0.895, 0.29], [0.29, 0.28]]
 *  - l2_w[2,1] = [[0.79], [1.8005]]
 * 
 * Forward 2:
 *  - l1_o[1,2] = [[0.7375, 0.425]]
 *  - l2_o[1,1] = [1.3478]
 * 
 * ```
 */
model_t* dummy_model() {
    model_t* mdl = model_new("dummy_model");
    graph_add_input(mdl->graph, 2, (int[]){1, 2}, false);
    node_t* l1 = graph_add_linear(mdl->graph, 2, true, NULL);
    node_t* l2 = graph_add_linear(mdl->graph, 1, true, NULL);
    
    // Init Params
    tensor_apply(l1->in[1], (float[]){1.0, 0.5, 0.5, 0.7}   , 4 * sizeof(float));
    tensor_apply(l2->in[1], (float[]){1.0, 2.0}             , 2 * sizeof(float));
    return mdl;
}

UnitTest_fn_def(test_dummy_create) {
    // Model
    tensor_t* ts, *x, *y;
    model_t* mdl = dummy_model();
    graph_dump(mdl->graph);
    model_show_tensors(mdl);

    // Train
    trainer_t* trn = trainer_new(0.1, 1e-8, TRAINER_LOSS_MSE, TRAINER_OPT_SGD);
    x = tensor_new_float32("x", (int[]){1, 2}, 2, (float[]){0.5, 1.0} , 2);
    y = tensor_new_float32("y", (int[]){1, 1}, 2, (float[]){0.8}      , 1);
    model_set_tensor(mdl, "Input0", x);
    float loss1 = trainer_step(trn, mdl, y);
    fprintf(stderr, "Loss: %f\n", loss1);
    model_eval(mdl, x);
    float loss2 = trainer_loss(trn, mdl, y, true);
    fprintf(stderr, "Loss: %f\n", loss2);

    ts = model_get_tensor(mdl, "Gemm1_out0");
    tensor_dump2(ts);


    return NULL;
}


UnitTest_fn_def(test_all) {
    device_reg("cpu");
    // UnitTest_add(test_mnist_create);
    UnitTest_add(test_simple_create);
    // UnitTest_add(test_dummy_create);
    return NULL;
}

UnitTest_run(test_all);