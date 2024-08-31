#include "sob.h"
#include <evo.h>


UnitTest_fn_def(test_model_create) {
    model_t* mdl = model_new("my_model");
    graph_dump(mdl->graph);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);