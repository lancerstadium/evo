#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_lut3d) {

    // runtime_t *rt = runtime_new("onnx");
    // runtime_load(rt, "model/lut3d_96/model.onnx");
    device_reg("cpu");
    serializer_t * sez = serializer_new("onnx");
    model_t * mdl = sez->load_model(sez, "model/lut3d_96/model.onnx");
    UnitTest_msg("load: %u", mdl->model_size);

    // runtime_run(rt);
    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_dump(mdl->graph);
    // graph_exec_report_level(rt->mdl->graph, 1); // Exec dump
    
    // runtime_unload(rt);
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