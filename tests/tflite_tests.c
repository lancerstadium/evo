#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_tflite_load) {
    device_reg("cpu");
    serializer_t * sez = serializer_new("tflite");
    context_t * ctx = sez->load_model(sez, "model/mnist_8/mnist_dw_f.tflite");

    graph_dump(ctx->graph->sub_vec[0]);

    ctx->sez->unload(ctx);
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_tflite_load);
    return NULL;
}

UnitTest_run(test_all);