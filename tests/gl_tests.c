#include "sob.h"
#include <evo.h>

canvas_t* my_cav;

UnitTest_fn_def(test_create_canvas) {
    // my_cav = canvas_new(720, 480);
    image_t * img = image_load("picture/input_0.jpg");
    UnitTest_msg("%s", image_dump_shape(img));
    my_cav = canvas_from_image(img);
    return NULL;
}

UnitTest_fn_def(test_fill_canvas) {
    // canvas_fill(my_cav, 0xffffffff);
    return NULL;
}

UnitTest_fn_def(test_line_canvas) {
    canvas_line(my_cav, 40, 60, 80, 360, 0xffffff00);
    canvas_line(my_cav, 40, 60, 20, 240, 0xffffff00);
    return NULL;
}

UnitTest_fn_def(test_rect_canvas) {
    canvas_rectangle(my_cav, 120, 30, 20, 40, 0xff00ff00);
    canvas_frame(my_cav, 120, 90, 20, 40, 1, 0xff00ff00);
    canvas_rectangle_c2(my_cav, 200, 150, 40, 60, 0xffff0000, 0xff0000ff);
    return NULL;
}

UnitTest_fn_def(test_circle_canvas) {
    canvas_ellipse(my_cav, 140, 180, 40, 20, 0xff00ff00);
    canvas_circle(my_cav, 140, 250, 40, 0xff0000ff);
    return NULL;
}

UnitTest_fn_def(test_text_canvas) {
    canvas_text(my_cav, "39.67%", 250, 250, &default_font, 2, 0xffc9b132);
    return NULL;
}

UnitTest_fn_def(test_export_canvas) {
    UnitTest_msg("%s", image_dump_shape(my_cav->background));
    canvas_export(my_cav, "cav.jpg");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_create_canvas);
    UnitTest_add(test_fill_canvas);
    UnitTest_add(test_line_canvas);
    UnitTest_add(test_rect_canvas);
    UnitTest_add(test_circle_canvas);
    UnitTest_add(test_text_canvas);
    UnitTest_add(test_export_canvas);
    return NULL;
}

UnitTest_run(test_all);