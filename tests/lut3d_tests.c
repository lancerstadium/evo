#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_lut3d) {

    runtime_t *rt = runtime_new("onnx");
    runtime_load(rt, "model/lut3d_96/model.onnx");

    // graph_dump(rt->mdl->graph);
    graph_prerun(rt->mdl->graph);
    graph_run(rt->mdl->graph);
    graph_exec_report_level(rt->mdl->graph, 1); // Exec dump
    
    
    runtime_free(rt);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_lut3d);
    return NULL;
}

UnitTest_run(test_all);