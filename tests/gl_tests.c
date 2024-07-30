#include "sob.h"
#include <evo.h>

canvas_t* my_cav;

UnitTest_fn_def(test_create_canvas) {
    my_cav = canvas_new(480, 480);
    return NULL;
}

UnitTest_fn_def(test_fill_canvas) {
    canvas_fill(my_cav, 0xffffffff);
    return NULL;
}

UnitTest_fn_def(test_line_canvas) {
    canvas_line(my_cav, 40, 60, 80, 360, 0xff000000);
    canvas_line(my_cav, 40, 60, 20, 240, 0xff000000);
    return NULL;
}

UnitTest_fn_def(test_export_canvas) {
    canvas_export(my_cav, "cav.png");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_create_canvas);
    UnitTest_add(test_fill_canvas);
    UnitTest_add(test_line_canvas);
    UnitTest_add(test_export_canvas);
    return NULL;
}

UnitTest_run(test_all);