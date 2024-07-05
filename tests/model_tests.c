
#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_model_load) {
    serializer_t * sez = serializer_new();
    context_t * ctx = sez->load_file(sez, "model/mnist_8/model.onnx");
    UnitTest_msg("load: %u", ctx->model_size);
    ctx->sez->unload(ctx);
    tensor_t * t = context_get_tensor(ctx, "Input3");
    tensor_dump(t);
    UnitTest_msg("unload: %u", ctx->model_size);
    serializer_free(sez);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_load);
    return NULL;
}

UnitTest_run(test_all);