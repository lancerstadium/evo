#include "sob.h"
#include <evo.h>

model_t* alexnet_model() {
    model_t* mdl = model_new("alexnet_model");
    graph_add_input(mdl->graph, 4, (int[]){1, 4, 224, 224}, TENSOR_TYPE_FLOAT32);
    graph_add_conv2d(mdl->graph, 6, (int64_t[]){11, 11}, (int64_t[]){4, 4}, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, (int64_t[]){2, 2}, NULL, NULL, 0, 0);
    // graph_add_conv2d(mdl->graph, 256, (int64_t[]){5, 5}, NULL, (int64_t[]){2, 2, 2, 2}, NULL, 1, NULL, "relu");
    // graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, (int64_t[]){2, 2}, NULL, NULL, 0, 0);
    // graph_add_conv2d(mdl->graph, 384, (int64_t[]){3, 3}, NULL, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    // graph_add_conv2d(mdl->graph, 384, (int64_t[]){3, 3}, NULL, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    // graph_add_conv2d(mdl->graph, 256, (int64_t[]){3, 3}, NULL, (int64_t[]){1, 1, 1, 1}, NULL, 1, NULL, "relu");
    // graph_add_maxpool2d(mdl->graph, (int64_t[]){3, 3}, (int64_t[]){2, 2}, NULL, NULL, 0, 0);
    graph_add_flatten(mdl->graph);
    // graph_add_linear(mdl->graph, 4096, true, "relu");
    // graph_add_dropout(mdl->graph, 0.5);
    // graph_add_linear(mdl->graph, 4096, true, "relu");
    // graph_add_dropout(mdl->graph, 0.5);
    graph_add_linear(mdl->graph, 10, true, "softmax");
    return mdl;
}

UnitTest_fn_def(test_tflite_load) {
    device_reg("cpu");
    // serializer_t * sez = serializer_get("onnx");
    // model_t * mdl = sez->load_file(sez, "model/mobilenet_v2_7/model.onnx");
    model_t* mdl = alexnet_model();

    graph_prerun(mdl->graph);

    graph_dump1(mdl->graph->sub_vec[0]);

    model_save(mdl, "wuhu.etm");

    // mdl->sez->unload(mdl);

    // serializer_t* esez = serializer_get("etm");
    // model_t* emdl = esez->load_file(esez, "wuhu.etm");
    // graph_dump(mdl->graph->sub_vec[0]);

    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_tflite_load);
    return NULL;
}

UnitTest_run(test_all);