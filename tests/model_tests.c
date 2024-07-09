#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_model_load) {
    device_t *cpu = device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    context_t * ctx = sez->load_model(sez, "model/resnet_18_v1_7/model.onnx");
    UnitTest_msg("load: %u", ctx->model_size);
    
    tensor_t * t1 = context_get_tensor(ctx, "data");
    tensor_dump(t1);
    tensor_t * t2 = context_get_tensor(ctx, "resnetv15_dense0_fwd");
    tensor_dump(t2);

    tensor_t * t3 = sez->load_tensor("model/mnist_8/test_data_set_0/input_0.pb");
    tensor_dump(t3);
    
    graph_t * sub_g = graph_sub(ctx->graph);
    graph_prerun(ctx->graph);
    graph_run(ctx->graph);
    graph_dump(ctx->graph); // Exec dump

    ctx->sez->unload(ctx);
    UnitTest_msg("unload: %u", ctx->model_size);
    serializer_free(sez);
    device_unreg_dev(cpu);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_load);
    return NULL;
}

UnitTest_run(test_all);