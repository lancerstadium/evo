#include <evo.h>
#include "sob.h"


UnitTest_fn_def(test_all) {
    figure_t* fig = figure_new("My figure", FIGURE_TYPE_VECTOR, 600, 400);
    figure_save(fig, "figure.svg");
    figure_free(fig);
    return NULL;
}



UnitTest_run(test_all);