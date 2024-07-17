#include "sob.h"
#include <evo.h>

int map_print(const void* key, size_t ksize, uintptr_t value, void* usr) {
    Log_info("name: %s", (char*)key);
}

UnitTest_fn_def(test_model_load) {
    device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    context_t * ctx = sez->load_model(sez, "model/mnist_8/model.onnx");
    UnitTest_msg("load: %u", ctx->model_size);
    
    tensor_t * t1 = context_get_tensor(ctx, "Input3");
    tensor_dump(t1);
    tensor_t * t2 = context_get_tensor(ctx, "Plus214_Output_0");
    tensor_dump(t2);

    tensor_t * t3 = sez->load_tensor("model/mnist_8/test_data_set_0/input_0.pb");
    tensor_dump2(t3);
    // hashmap_iterate(ctx->tensor_map, map_print, NULL);

    tensor_copy(t1, t3);
    printf("ndata1: %ld, ndata3: %ld\n", t1->ndata, t3->ndata);
    tensor_dump2(t1);

    graph_prerun(ctx->graph);
    graph_run(ctx->graph);
    graph_dump(ctx->graph); // Exec dump

    tensor_dump2(t2);

    ctx->sez->unload(ctx);
    UnitTest_msg("unload: %u", ctx->model_size);
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_load);
    return NULL;
}

UnitTest_run(test_all);