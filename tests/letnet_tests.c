#include "sob.h"
#include <evo.h>

/**
 * ref: https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
 * 
 * 
 */
model_t* letnet_model() {
    model_t* mdl = model_new("letnet_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28}, TENSOR_TYPE_FLOAT32);
    graph_add_conv2d(mdl->graph, 6, (int64_t[]){5, 5}, NULL, (int64_t[]){2, 2, 2, 2}, NULL, 1, NULL, "tanh");
    graph_add_avgpool2d(mdl->graph, (int64_t[]){2, 2}, (int64_t[]){2, 2}, NULL, 0);
    graph_add_conv2d(mdl->graph, 16, (int64_t[]){5, 5}, NULL, NULL, NULL, 1, NULL, "tanh");
    graph_add_avgpool2d(mdl->graph, (int64_t[]){2, 2}, (int64_t[]){2, 2}, NULL, 0);
    graph_add_flatten(mdl->graph);
    graph_add_linear(mdl->graph, 120, true, "tanh");
    graph_add_linear(mdl->graph, 84, true, "tanh");
    graph_add_linear(mdl->graph, 10, true, "softmax");
    return mdl;
}


UnitTest_fn_def(test_letnet) {

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
    model_t* mdl = letnet_model();
    graph_dump(mdl->graph);
    model_show_tensors(mdl);

    // Train
    int nepoch = 10;
    int nbatch = 60000;
    trainer_t* trn = trainer_new(0.001, 1e-8, TRAINER_LOSS_CROSS_ENTROPY, TRAINER_OPT_SGD);
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
    figure_t* fig = figure_new_1d("LeNet Loss", FIGURE_TYPE_VECTOR, FIGURE_PLOT_TYPE_LINE, loss_vec);
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
            trainer_step(trn, mdl, y_ts);
            // if(e == 0 && b >= 0 && b < 1) {
            //     // fprintf(stderr, "<%u> ", label->bs[b]);
            //     // image_dump_raw(imgs, b);
            //     // tensor_dump1(y_ts);
            //     ts = model_get_tensor(mdl, "Flatten6_out0");
            //     tensor_dump1(ts);
            //     ts = model_get_tensor(mdl, "Gemm11_out0");
            //     tensor_dump1(ts);
            //     fprintf(stderr, "--\n");
            // }
            trainer_zero_grad(trn, mdl);
            // tensor_t* sss = model_get_tensor(mdl, "Gemm1_out0");
            // tensor_dump2(sss);
            sum_loss += trn->cur_loss;
        }
        loss_data[e] = sum_loss / nbatch;
        char bar_label[50];
        sprintf(bar_label, "Train: %d/%d Loss: %.6f", e, nepoch, loss_data[e]);
        progressbar_update_label(bar, bar_label);
        progressbar_update(bar, e);
        figure_update_plot_1d(fig, loss_vec, e);
        figure_save(fig, "loss.svg");
    }
    progressbar_finish(bar);
    progressbar_free(bar);
    figure_free(fig);


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

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_letnet);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);