
#include "sob.h"
#include <evo/evo.h>

UnitTest_fn_def(test_node_init) {
    context_t * ctx = context_new("nihao");
    graph_t * g = graph_new(ctx);
    node_t* nd = node_new(g, "nd1", OP_TYPE_ABS);
    UnitTest_msg("%s", nd->name);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_node_init);
    return NULL;
}

UnitTest_run(test_all);