
#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_node_init) {
    context_t * ctx = context_new("nihao");
    graph_t * g = graph_new(ctx);
    node_t* nd = node_new(g, "nd1", OP_TYPE_ABS);
    UnitTest_msg("%s", nd->name);
    return NULL;
}

UnitTest_fn_def(test_matmul_2d) {
    device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    context_t * ctx = sez->load_model(sez, "node/test_matmul_2d/model.onnx");

    tensor_t * t0 = sez->load_tensor("node/test_matmul_2d/test_data_set_0/input_0.pb");
    tensor_dump2(t0);
    tensor_t * t1 = sez->load_tensor("node/test_matmul_2d/test_data_set_0/input_1.pb");
    tensor_dump2(t1);


    ctx->sez->unload(ctx);
    serializer_free(sez);
    device_unreg("cpu");
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_node_init);
    UnitTest_add(test_matmul_2d);
    return NULL;
}

UnitTest_run(test_all);