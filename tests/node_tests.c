
#include "sob.h"
#include <evo.h>

serializer_t * sez;

UnitTest_fn_def(test_node_init) {
    device_reg("cpu");
    sez = serializer_new("onnx");
    return NULL;
}

UnitTest_fn_def(test_add) {
    context_t * ctx = sez->load_model(sez, "node/test_add/model.onnx");

    tensor_t * a = context_get_tensor(ctx, "x");
    tensor_t * b = context_get_tensor(ctx, "y");
    tensor_t * c = context_get_tensor(ctx, "sum");

    tensor_t * t0 = sez->load_tensor("node/test_add/test_data_set_0/input_0.pb");
    tensor_t * t1 = sez->load_tensor("node/test_add/test_data_set_0/input_1.pb");
    tensor_t * t3 = sez->load_tensor("node/test_add/test_data_set_0/output_0.pb");

    tensor_copy(a, t0); 
    tensor_copy(b, t1);

    graph_prerun(ctx->graph);
    graph_run(ctx->graph);

    tensor_dump2(c);    /** TODO: fixup add */
    // UnitTest_ast(tensor_equal(c, t3), "test_add failed");
    
    ctx->sez->unload(ctx);
    return NULL;
}

UnitTest_fn_def(test_matmul_2d) {
    context_t * ctx = sez->load_model(sez, "node/test_matmul_2d/model.onnx");

    tensor_t * a = context_get_tensor(ctx, "a");
    tensor_t * b = context_get_tensor(ctx, "b");
    tensor_t * c = context_get_tensor(ctx, "c");

    tensor_t * t0 = sez->load_tensor("node/test_matmul_2d/test_data_set_0/input_0.pb");
    tensor_t * t1 = sez->load_tensor("node/test_matmul_2d/test_data_set_0/input_1.pb");
    tensor_t * t3 = sez->load_tensor("node/test_matmul_2d/test_data_set_0/output_0.pb");

    tensor_copy(a, t0); 
    tensor_copy(b, t1);

    graph_prerun(ctx->graph);
    graph_run(ctx->graph);

    UnitTest_ast(tensor_equal(c, t3), "test_matmul_2d failed");

    ctx->sez->unload(ctx);
    return NULL;
}


UnitTest_fn_def(test_node_exit) {
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_node_init);
    UnitTest_add(test_add);
    UnitTest_add(test_matmul_2d);
    UnitTest_add(test_node_exit);
    return NULL;
}

UnitTest_run(test_all);