#include "sob.h"
#include <evo.h>

/**
 * ref: https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
 * 
 * 
 */
model_t* letnet_model() {
    model_t* mdl = model_new("letnet_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28}, false);
    graph_add_conv2d(mdl->graph, 6, (int64_t[]){5, 5}, NULL, (int64_t[]){2, 2, 2, 2}, NULL, 1, NULL, "tanh");
    graph_add_avgpool2d(mdl->graph, (int64_t[]){2, 2}, (int64_t[]){2, 2}, NULL, 0);
    graph_add_conv2d(mdl->graph, 16, (int64_t[]){5, 5}, NULL, NULL, NULL, 1, NULL, "tanh");
    graph_add_avgpool2d(mdl->graph, (int64_t[]){2, 2}, (int64_t[]){2, 2}, NULL, 0);
    graph_add_flatten(mdl->graph);
    graph_add_linear(mdl->graph, 120, true, "tanh");
    graph_add_linear(mdl->graph, 84, true, "tanh");
    graph_add_linear(mdl->graph, 10, true, NULL);
    return mdl;
}


UnitTest_fn_def(test_letnet) {

    model_t* mdl = letnet_model();
    graph_dump(mdl->graph);

    return NULL;
}

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_letnet);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);