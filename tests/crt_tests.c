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
 * 
 * ref: https://blog.csdn.net/weixin_45954454/article/details/114455209
 * 
 *  ** Mnist Model ** :
 *      - Flatten           : [1,1,28,28] ->     [1,784]
 *      - FC                :     [1,784] ->     [1,128]
 *      - ReLU              :     [1,128] ->     [1,128]
 *      - FC                :     [1,128] ->      [1,10]
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
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28}, TENSOR_TYPE_FLOAT32);
    graph_add_flatten(mdl->graph);
    graph_add_linear(mdl->graph, 128, true, "tanh");
    graph_add_linear(mdl->graph, 10, true, "softmax");
    return mdl;
}

UnitTest_fn_def(test_mnist_create) {
    // Train Dataset
    srand((unsigned)time(NULL));
    const char* image_filename = "picture/mnist/train-images-idx3-ubyte";
    const char* label_filename = "picture/mnist/train-labels-idx1-ubyte";
    image_t* imgs = image_load_mnist(image_filename, label_filename);
    if(!imgs) {
        fprintf(stderr, "Load mnist fail, please exec `download_mnist.sh` in Dir `picture`.\n");
        return "Load Mnist Fail!";
    } else {
        fprintf(stderr, "Load Mnist Success!\n");
    }
    attribute_t* label = image_get_attr(imgs, "label");
    tensor_t* imgs_f32 = tensor_cast(imgs->raw, TENSOR_TYPE_FLOAT32);
    for(int i = 0; i < imgs_f32->ndata; i++) {
        ((float*)(imgs_f32->datas))[i] /= 255.0f;
    }

    // Model
    model_t* mdl = mnist_model();
    graph_dump(mdl->graph);
    model_show_tensors(mdl);

    // Train
    int nepoch = 10;
    int nbatch = 60000;
    optimizer_t* opt = optimizer_new(0.001, 1e-8, OPTIMIZER_LOSS_TYPE_CROSS_ENTROPY, OPTIMIZER_TYPE_SGD);
    tensor_t *x_ts = tensor_new("x", TENSOR_TYPE_FLOAT32);
    tensor_t *y_ts = tensor_new("y", TENSOR_TYPE_FLOAT32);
    tensor_t *ts;
    tensor_t* loss_vec = tensor_new("loss", TENSOR_TYPE_FLOAT32);
    tensor_reshape(x_ts     , 4, (int[]){1, 1, 28, 28});
    tensor_reshape(y_ts     , 2, (int[]){1, 10});
    tensor_reshape(loss_vec , 2, (int[]){nepoch, 1});
    float* xd = x_ts->datas;
    float* yd = y_ts->datas;
    float* loss_data = loss_vec->datas;

    progressbar_t *bar = progressbar_new_format("Train:", nepoch, "[=]");
    figure_t* fig = figure_new_1d("Mnist Loss", FIGURE_TYPE_VECTOR, FIGURE_PLOT_TYPE_LINE, loss_vec);
    fig->axiss[1]->is_auto_scale = false;
    fig->axiss[1]->range_min = -0.01;
    fig->axiss[1]->range_max = 0.1;
    for (int e = 0; e < nepoch; e++) {
        // Mini-batch training
        float sum_loss = 0.0f;
        for (int b = 0; b < nbatch; b++) {
            tensor_apply(x_ts, imgs_f32->datas + b * 784 * sizeof(float), 784 * sizeof(float));
            model_set_tensor(mdl, "Input0", x_ts);
            tensor_fill_zero(y_ts);
            ((float*)y_ts->datas)[label->bs[b]] = 1;
            optimizer_step(opt, mdl, y_ts);
            // if(e == 3 && b >= 0 && b < 1) {
            //     fprintf(stderr, "<%u> ", label->bs[b]);
            //     // image_dump_raw(imgs, b);
            //     tensor_dump1(y_ts);
            //     ts = model_get_tensor(mdl, "Softmax4_out0"); tensor_dump1(ts);
            //     tensor_dump1(ts->grad);
            //     ts = model_get_tensor(mdl, "Gemm3_out0");
            //     tensor_dump1(ts->grad);
            //     fprintf(stderr, "--\n");
            // }
            optimizer_zero_grad(opt, mdl);
            // tensor_t* sss = model_get_tensor(mdl, "Gemm1_out0");
            // tensor_dump2(sss);
            sum_loss += opt->cur_loss;
        }
        loss_data[e] = sum_loss / nbatch;
        // Evaluate the model on the training and test set
        if (e % 2 == 0) {

        }
        char bar_label[50];
        sprintf(bar_label, "Loss: %.6f", loss_data[e]);
        progressbar_update_label(bar, bar_label);
        progressbar_update(bar, e);
        figure_update_plot_1d(fig, loss_vec, e);
        figure_save(fig, "loss.svg");
    }
    progressbar_finish(bar);
    progressbar_free(bar);
    figure_free(fig);

    // Paint

    // Eval
    float train_acc = 0.0;
    int acc_cnt = 0;
    for(int b = 0; b < nbatch; b++) {
        tensor_apply(x_ts, imgs_f32->datas + b * 784 * sizeof(float), 784 * sizeof(float));
        model_set_tensor(mdl, "Input0", x_ts);
        tensor_t* y_res = model_eval(mdl, x_ts);
        tensor_t* y_us = tensor_squeeze(y_res, NULL, 0);
        tensor_t* y_out = tensor_argmax(y_us, 0, 1, 0);
        int64_t yy = ((int64_t*)y_out->datas)[0];
        acc_cnt += ((yy == (int64_t)label->bs[b]) ? 1 : 0);
        // fprintf(stderr, "<%u %ld> ", label->bs[b], yy);
        // tensor_dump2(y_out);
    }
    train_acc += ((float)acc_cnt / (float)nbatch);

    // Test Dataset
    const char* image_test_filename = "picture/mnist/t10k-images-idx3-ubyte";
    const char* label_test_filename = "picture/mnist/t10k-labels-idx1-ubyte";
    image_t* imgs_test = image_load_mnist(image_test_filename, label_test_filename);
    if(!imgs_test) {
        fprintf(stderr, "Load mnist fail, please exec `download_mnist.sh` in Dir `picture`.\n");
        return "Load Mnist Fail!";
    } else {
        fprintf(stderr, "Load Mnist Success!\n");
    }
    attribute_t* label_test = image_get_attr(imgs_test, "label");
    tensor_t* imgs_test_f32 = tensor_cast(imgs_test->raw, TENSOR_TYPE_FLOAT32);
    for(int i = 0; i < imgs_test_f32->ndata; i++) {
        ((float*)(imgs_test_f32->datas))[i] /= 255.0f;
    }

    float test_acc = 0.0;
    acc_cnt = 0;
    nbatch = 10000;
    for(int b = 0; b < nbatch; b++) {
        tensor_apply(x_ts, imgs_test_f32->datas + b * 784 * sizeof(float), 784 * sizeof(float));
        model_set_tensor(mdl, "Input0", x_ts);
        tensor_t* y_res = model_eval(mdl, x_ts);
        tensor_t* y_us = tensor_squeeze(y_res, NULL, 0);
        tensor_t* y_out = tensor_argmax(y_us, 0, 1, 0);
        int64_t yy = ((int64_t*)y_out->datas)[0];
        acc_cnt += ((yy == (int64_t)label_test->bs[b]) ? 1 : 0);
        if(b < 10) {
            fprintf(stderr, "<%u %ld> ", label_test->bs[b], yy);
            // tensor_dump2(y_out);
        }
    }
    test_acc += ((float)acc_cnt / (float)nbatch);

    // Summary
    fprintf(stderr, "Train acc: %.2f%%, Test acc: %.2f%%\n", train_acc * 100, test_acc * 100);

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
    graph_add_input(mdl->graph, 2, (int[]){1, 2}, TENSOR_TYPE_FLOAT32);
    node_t* l1 = graph_add_linear(mdl->graph, 3, true, "tanh");
    node_t* l2 = graph_add_linear(mdl->graph, 1, true, NULL);

    // Init Param
    // tensor_apply(l1->in[1], (float[]){0.1, 0.2, 0.3, 0.4, 0.5, 0.6}     , 6 * sizeof(float));
    // tensor_apply(l1->in[2], (float[]){0.01, 0.02, 0.03}                 , 3 * sizeof(float));
    // tensor_apply(l2->in[1], (float[]){0.1, 0.2, 0.3}                    , 3 * sizeof(float));
    // tensor_apply(l2->in[2], (float[]){0.05}                             , 1 * sizeof(float));
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
    int nepoch = 15000;
    tensor_t* loss_vec = tensor_new("loss", TENSOR_TYPE_FLOAT32);
    tensor_reshape(loss_vec, 2, (int[]){nepoch, 1});
    float* loss_data = loss_vec->datas;
    optimizer_t* opt = optimizer_new(0.001, 1e-8, OPTIMIZER_LOSS_TYPE_MSE, OPTIMIZER_TYPE_SGD);
    tensor_t* X_ts, *y_ts;
    for(int e = 0; e < nepoch; e++) {
        for(int b = 0; b < sizeof(y)/sizeof(float); b++) {
            X_ts = tensor_new_float32("X", (int[]){1, X_off}, 2, X + b * X_off, X_off);
            y_ts = tensor_new_float32("y", (int[]){1, y_off}, 2, y + b, y_off);
            model_set_tensor(mdl, "Input0", X_ts);
            optimizer_step(opt, mdl, y_ts);
            optimizer_zero_grad(opt, mdl);
            // model_eval(mdl, X_ts);
            // fprintf(stderr, "<%.0f %2.2f> ", y[b], opt->cur_loss);
        }
        // fprintf(stderr, "[%2d] Loss: %.8f\n", e, opt->cur_loss);
        loss_data[e] = opt->cur_loss;
    }

    figure_t* fig = figure_new_1d("Simple Loss", FIGURE_TYPE_VECTOR, FIGURE_PLOT_TYPE_LINE, loss_vec);
    fig->axiss[1]->is_auto_scale = false;
    fig->axiss[1]->range_min = -0.01;
    fig->axiss[1]->range_max = 0.2;
    figure_save(fig, "loss.svg");

    // Eval
    for(int b = 0; b < sizeof(y)/sizeof(float); b++) {
        X_ts = tensor_new_float32("X", (int[]){1, X_off}, 2, X + b * X_off, X_off);
        y_ts = model_eval(mdl, X_ts);
        fprintf(stderr, "<%f %f> ", y[b], y_ts->datas ? ((float*)y_ts->datas)[0] : 0.0f);
        tensor_t* sss = model_get_tensor(mdl, "Gemm2_out0");
        // tensor_dump2(sss);
    }
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
    graph_add_input(mdl->graph, 2, (int[]){1, 2}, TENSOR_TYPE_FLOAT32);
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
    optimizer_t* opt = optimizer_new(0.1, 1e-8, OPTIMIZER_LOSS_TYPE_MSE, OPTIMIZER_TYPE_SGD);
    x = tensor_new_float32("x", (int[]){1, 2}, 2, (float[]){0.5, 1.0} , 2);
    y = tensor_new_float32("y", (int[]){1, 1}, 2, (float[]){0.8}      , 1);
    model_set_tensor(mdl, "Input0", x);
    float loss1 = optimizer_step(opt, mdl, y);
    fprintf(stderr, "Loss: %f\n", loss1);
    model_eval(mdl, x);
    float loss2 = optimizer_loss(opt, mdl, y, true);
    fprintf(stderr, "Loss: %f\n", loss2);

    ts = model_get_tensor(mdl, "Gemm1_out0");
    tensor_dump2(ts);


    return NULL;
}


UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_mnist_create);
    // UnitTest_add(test_simple_create);
    // UnitTest_add(test_dummy_create);
    return NULL;
}

UnitTest_run(test_all);