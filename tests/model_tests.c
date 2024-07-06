
#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_model_load) {
    serializer_t * sez = serializer_new("onnx");
    context_t * ctx = sez->load_file(sez, "model/mnist_8/model.onnx");
    UnitTest_msg("load: %u", ctx->model_size);
    ctx->sez->unload(ctx);
    tensor_t * t1 = context_get_tensor(ctx, "Input3");
    tensor_dump(t1);
    tensor_t * t2 = context_get_tensor(ctx, "Plus214_Output_0");
    tensor_dump(t2);
    UnitTest_msg("nnode: %u", ctx->graph->nnode);
    UnitTest_msg("unload: %u", ctx->model_size);
    node_dump(ctx->graph->nodes[1]);
    serializer_free(sez);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_load);
    return NULL;
}

UnitTest_run(test_all);