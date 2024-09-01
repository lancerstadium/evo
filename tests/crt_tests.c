#include "sob.h"
#include <evo.h>

model_t* mnist_model_def() {
    model_t* mdl = model_new("mnist_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28});
    graph_add_flatten(mdl->graph);
    graph_add_dense(mdl->graph, 128, "relu");
    graph_add_dense(mdl->graph, 10, "softmax");
    return mdl;
}

UnitTest_fn_def(test_model_create) {
    device_reg("cpu");
    model_t* mdl = mnist_model_def();
    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_posrun(mdl->graph);
    graph_dump(mdl->graph);
    model_dump_tensor(mdl);
    tensor_t* ts3 = model_get_tensor(mdl, "Softmax4_out0");
    tensor_dump2(ts3);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);