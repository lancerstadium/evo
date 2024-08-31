#include "sob.h"
#include <evo.h>



UnitTest_fn_def(test_model_create) {
    device_reg("cpu");
    model_t* mdl = model_new("my_model");
    tensor_t* ts1 = tensor_new_float32("add01_in1", (int[]){2, 3}, 2, (float[]){1, 2, 3, 3, 2, 1}, 6);
    tensor_t* ts2 = tensor_new_float32("add01_in2", (int[]){3, 2}, 2, (float[]){2, 2, 2, 3, 3, 3}, 6);
    graph_add_layer(mdl->graph, OP_TYPE_MAT_MUL, (tensor_t*[]){ts1, ts2}, 2, 1);
    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_posrun(mdl->graph);
    graph_dump2(mdl->graph);
    model_dump_tensor(mdl);
    tensor_t* ts3 = model_get_tensor(mdl, "MatMul0_out0");
    tensor_dump2(ts3);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);