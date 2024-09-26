#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_tflite_load) {
    device_reg("cpu");
    serializer_t * sez = serializer_get("onnx");
    model_t * mdl = sez->load_file(sez, "model/mnist_8/model.onnx");

    graph_prerun(mdl->graph);

    graph_dump2(mdl->graph->sub_vec[0]);

    model_save(mdl, "wuhu.etm");

    mdl->sez->unload(mdl);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_tflite_load);
    return NULL;
}

UnitTest_run(test_all);