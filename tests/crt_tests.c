#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_model_create) {
    device_reg("cpu");
    model_t* mdl = model_new("my_model");
    node_t* nd = node_new(mdl->graph, "add01", OP_TYPE_ADD);
    graph_push_node(mdl->graph, nd);
    graph_dump(mdl->graph);
    model_dump_tensor(mdl);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);