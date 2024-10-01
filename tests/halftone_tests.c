#include "sob.h"
#include <evo.h>

#define WIDTH 720
#define HEIGHT 720
#define BG_COLOR 0xFF181818


canvas_t* cat_cav = NULL;
canvas_t* kitten(canvas_t* cav, float dt) {
    if(!cav) return NULL;

    // 0. init canvas
    canvas_fill(cav, BG_COLOR);
    if(dt == 0.0) {
        image_t* cat = image_load("picture/kitten.jpg");
        image_to_grey(cat);
        image_dithering(cat);
        cat_cav = canvas_from_image(cat);
    }


    // 1. cat photo
    {
        canvas_draw(cav, 0, 0, cat_cav->width, cat_cav->height, cat_cav->pixels);
    }

    return cav;
}

UnitTest_fn_def(test_floyd_steinberg) {
    // image_to_grey(img);
    renderer_t* rd = renderer_new(WIDTH, HEIGHT, RENDERER_TYPE_LINUX);
    renderer_run(rd, kitten);
    renderer_free(rd);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_floyd_steinberg);
    return NULL;
}

UnitTest_run(test_all);