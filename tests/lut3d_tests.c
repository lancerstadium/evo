#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_lut3d) {
    device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    model_t * mdl = sez->load_model(sez, "model/lut3d_96/model.onnx");
    UnitTest_msg("load: %u", mdl->model_size);

    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_exec_report_level(mdl->graph, 1); // Exec dump

    mdl->sez->unload(mdl);
    UnitTest_msg("unload: %u", mdl->model_size);
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_lut3d);
    return NULL;
}

UnitTest_run(test_all);