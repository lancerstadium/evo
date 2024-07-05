
#include "sob.h"
#include <evo/evo.h>

UnitTest_fn_def(test_model_load) {
    serializer_t * sez = serializer_new();
    context_t * ctx = sez->load_file(sez, "model/mnist_8/model.onnx");
    UnitTest_msg("%u", ctx->model_size);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_load);
    return NULL;
}

UnitTest_run(test_all);