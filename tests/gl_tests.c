#include "sob.h"
#include <evo.h>

canvas_t* my_cav;

UnitTest_fn_def(test_create_canvas) {
    my_cav = canvas_new(480, 720);
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

UnitTest_fn_def(test_rect_canvas) {
    canvas_rectangle(my_cav, 120, 30, 20, 40, 0xff000000);
    canvas_frame(my_cav, 120, 90, 20, 40, 1, 0xff000000);
    return NULL;
}

UnitTest_fn_def(test_circle_canvas) {
    canvas_ellipse(my_cav, 140, 180, 40, 20, 0xff000000);
    canvas_circle(my_cav, 140, 250, 40, 0xff000000);
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
    UnitTest_add(test_rect_canvas);
    UnitTest_add(test_circle_canvas);
    UnitTest_add(test_export_canvas);
    return NULL;
}

UnitTest_run(test_all);