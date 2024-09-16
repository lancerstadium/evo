#include <evo.h>
#include "sob.h"


UnitTest_fn_def(test_all) {
    figure_t* fig = figure_new("ASDASD", FIGURE_TYPE_VECTOR, 720, 480, 2);
    figure_plot_t* p1 = figure_plot_new("wuhu", FIGURE_PLOT_TYPE_LINE, (float[]){1.2, -2.1, 3.4, 9.6, 11.7, 4.5}, 3, 2);
    p1->mtype = 'x';
    figure_plot_t* p2 = figure_plot_new("eihei", FIGURE_PLOT_TYPE_SCATTER, (float[]){2.3, 4.1, 4.9, -3.7, 8.1, 2.9}, 3, 2);
    p2->mtype = 'o';
    figure_set_xlabel(fig, "xx");
    figure_set_ylabel(fig, "yy");
    figure_add_plot(fig, p1);
    figure_add_plot(fig, p2);
    figure_save(fig, "figure.svg");
    figure_free(fig);
    return NULL;
}



UnitTest_run(test_all);