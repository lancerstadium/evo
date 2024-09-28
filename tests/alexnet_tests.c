#include "sob.h"
#include <evo.h>

/**
 * ref: https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html
 * 
 * 
 */
model_t* alexnet_model() {
    model_t* mdl = model_new("alexnet_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 4, 224, 224}, TENSOR_TYPE_FLOAT32);
    graph_add_conv2d(mdl->graph, 6, (int64_t[]){11, 11}, (int64_t[]){4, 4}, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, (int64_t[]){2, 2}, NULL, NULL, 0, 0);
    // graph_add_conv2d(mdl->graph, 256, (int64_t[]){5, 5}, NULL, (int64_t[]){2, 2, 2, 2}, NULL, 1, NULL, "relu");
    // graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, (int64_t[]){2, 2}, NULL, NULL, 0, 0);
    // graph_add_conv2d(mdl->graph, 384, (int64_t[]){3, 3}, NULL, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    // graph_add_conv2d(mdl->graph, 384, (int64_t[]){3, 3}, NULL, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    // graph_add_conv2d(mdl->graph, 256, (int64_t[]){3, 3}, NULL, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    // graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, (int64_t[]){2, 2}, NULL, NULL, 0, 0);
    graph_add_flatten(mdl->graph);
    // graph_add_linear(mdl->graph, 4096, true, "relu");
    // graph_add_dropout(mdl->graph, 0.5);
    // graph_add_linear(mdl->graph, 4096, true, "relu");
    // graph_add_dropout(mdl->graph, 0.5);
    graph_add_linear(mdl->graph, 10, true, "softmax");
    return mdl;
}


UnitTest_fn_def(test_alexnet) {

    model_t* mdl = alexnet_model();
    graph_dump1(mdl->graph);
    model_show_tensors(mdl);

    return NULL;
}

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_alexnet);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);