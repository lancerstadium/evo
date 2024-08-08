#include <evo.h>
#include <math.h>
#include "sob.h"

#define PI 3.14159265359

#define WIDTH 960
#define HEIGHT 720
#define BG_COLOR 0xFF181818
#define GRID_COUNT 10
#define GRID_PAD 0.5 / GRID_COUNT
#define GRID_SIZE ((GRID_COUNT - 1) * GRID_PAD)
#define CIRCLE_RADIUS 5
#define Z_START 0.25
#define ABOBA_PADDING 50

canvas_t* dots3d(float dt) {
    float angle = 0;
    angle += 0.25 * PI * dt;
    canvas_t* cav = canvas_new(WIDTH, HEIGHT);
    canvas_fill(cav, BG_COLOR);
    // 1. draw circles
    for (int ix = 0; ix < GRID_COUNT; ++ix) {
        for (int iy = 0; iy < GRID_COUNT; ++iy) {
            for (int iz = 0; iz < GRID_COUNT; ++iz) {
                float x = ix * GRID_PAD - GRID_SIZE / 2;
                float y = iy * GRID_PAD - GRID_SIZE / 2;
                float z = Z_START + iz * GRID_PAD;

                float cx = 0.0;
                float cz = Z_START + GRID_SIZE / 2;

                float dx = x - cx;
                float dz = z - cz;

                float a = atan2f(dz, dx);
                float m = sqrtf(dx * dx + dz * dz);

                dx = cosf(a + angle) * m;
                dz = sinf(a + angle) * m;

                x = dx + cx;
                z = dz + cz;

                x /= z;
                y /= z;

                uint32_t r = ix * 255 / GRID_COUNT;
                uint32_t g = iy * 255 / GRID_COUNT;
                uint32_t b = iz * 255 / GRID_COUNT;
                uint32_t color = pixel_rgba(r, g, b, 255);
                canvas_circle(cav, (x + 1) / 2 * WIDTH, (y + 1) / 2 * HEIGHT, CIRCLE_RADIUS, color);
            }
        }
    }
    return cav;
}


UnitTest_fn_def(test_render_canvas) {
    renderer_t* rd = renderer_new(RENDERER_TYPE_GIF);
    renderer_run(rd, dots3d);
    renderer_free(rd);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_render_canvas);
    return NULL;
}

UnitTest_run(test_all);