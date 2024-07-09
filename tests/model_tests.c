#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_model_load) {
    device_t *cpu = device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    context_t * ctx = sez->load_file(sez, "model/mnist_8/model.onnx");
    UnitTest_msg("load: %u", ctx->model_size);
    
    tensor_t * t1 = context_get_tensor(ctx, "Input3");
    tensor_dump(t1);
    tensor_t * t2 = context_get_tensor(ctx, "Plus214_Output_0");
    tensor_dump(t2);
    
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