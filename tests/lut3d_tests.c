#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_lut3d) {
    device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    context_t * ctx = sez->load_model(sez, "model/resnet_18_v1_7/model.onnx");
    UnitTest_msg("load: %u", ctx->model_size);

    graph_prerun(ctx->graph);
    graph_run(ctx->graph);
    if(ctx->graph)
        graph_dump(ctx->graph); // Exec dump

    ctx->sez->unload(ctx);
    UnitTest_msg("unload: %u", ctx->model_size);
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_lut3d);
    return NULL;
}

UnitTest_run(test_all);