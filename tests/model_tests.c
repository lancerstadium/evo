#include "sob.h"
#include <evo.h>

serializer_t * sez;

UnitTest_fn_def(test_model_init) {
    device_reg("cpu");
    sez = serializer_new("onnx");
    return NULL;
}

UnitTest_fn_def(test_mnist_8) {
    context_t * ctx = sez->load_model(sez, "model/mnist_8/model.onnx");
    tensor_t * x = context_get_tensor(ctx, "Input3");
    tensor_t * y = context_get_tensor(ctx, "Plus214_Output_0");
    tensor_t * t0 = sez->load_tensor("model/mnist_8/test_data_set_0/input_0.pb");

    tensor_copy(x, t0);
    graph_prerun(ctx->graph);
    graph_run(ctx->graph);
    graph_dump(ctx->graph);

    graph_exec_report(ctx->graph->sub_vec[0]);

    tensor_dump2(y);
    ctx->sez->unload(ctx);
    return NULL;
}

UnitTest_fn_def(test_model_exit) {
    serializer_free(sez);
    device_unreg("cpu");
}

// ---------------------- All    ----------------------

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_init);
    UnitTest_add(test_mnist_8);
    UnitTest_add(test_model_exit);
    return NULL;
}

UnitTest_run(test_all);