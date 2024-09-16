#include <evo.h>
#include "sob.h"


UnitTest_fn_def(test_all) {
    figure_t* fig = figure_new("My figure", FIGURE_TYPE_VECTOR, 600, 400, 2);
    figure_plot_t* p1 = figure_plot_new("wuhu", FIGURE_PLOT_TYPE_SCATTER, (float[]){1.2, -2.1, 3.4, 9.6, 11.7, 4.5}, 3, 2);
    p1->mtype = 'x';
    figure_axis_set_label(fig->axiss[0], "x");
    figure_axis_set_label(fig->axiss[1], "y");
    figure_add_plot(fig, p1);
    figure_save(fig, "figure.svg");
    figure_free(fig);
    return NULL;
}



UnitTest_run(test_all);