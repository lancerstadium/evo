#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_tflite_load) {
    device_reg("cpu");
    serializer_t * sez = serializer_new("tflite");
    model_t * mdl = sez->load_model(sez, "model/mnist_8/mnist_dw_q.tflite");

    graph_dump1(mdl->graph->sub_vec[0]);

    mdl->sez->unload(mdl);
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_tflite_load);
    return NULL;
}

UnitTest_run(test_all);