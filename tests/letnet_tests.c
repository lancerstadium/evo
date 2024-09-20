#include "sob.h"
#include <evo.h>

/**
 * ref: https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
 * 
 * 
 */
model_t* letnet_model() {
    model_t* mdl = model_new("mnist_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28}, false);
    graph_add_flatten(mdl->graph);
    graph_add_linear(mdl->graph, 128, true, "tanh");
    graph_add_linear(mdl->graph, 10, true, "softmax");
    return mdl;
}


UnitTest_fn_def(test_letnet) {


    return NULL;
}

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_letnet);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);