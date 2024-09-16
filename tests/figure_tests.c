#include <evo.h>
#include "sob.h"


UnitTest_fn_def(test_all) {
    figure_t* fig = figure_new("My figure", FIGURE_TYPE_VECTOR, 600, 400, 2);
    figure_axis_set_label(fig->axiss[0], "x");
    figure_axis_set_label(fig->axiss[1], "y");
    figure_save(fig, "figure.svg");
    figure_free(fig);
    return NULL;
}



UnitTest_run(test_all);