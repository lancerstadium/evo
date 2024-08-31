#include <evo.h>
#include <math.h>
#include "sob.h"

#define PI 3.14159265359

#define WIDTH 960
#define HEIGHT 720
#define BG_COLOR 0xFF181818

canvas_t* dots3d(canvas_t* cav, float dt) {
    // Parameters
    int grid_count = 10;
    int circle_radius = 5;
    float grid_pad = 0.5 / grid_count;
    float grid_size = (grid_count - 1) * grid_pad;
    float angle = 0;
    float z_start = 0.25;
    angle += 0.25 * PI * dt;
    // 0. init canvas
    canvas_fill(cav, BG_COLOR);
    // 1. draw circles
    for (int ix = 0; ix < grid_count; ++ix) {
        for (int iy = 0; iy < grid_count; ++iy) {
            for (int iz = 0; iz < grid_count; ++iz) {
                float x = ix * grid_pad - grid_size / 2;
                float y = iy * grid_pad - grid_size / 2;
                float z = z_start + iz * grid_pad;

                float cx = 0.0;
                float cz = z_start + grid_size / 2;

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

                uint32_t r = ix * 255 / grid_count;
                uint32_t g = iy * 255 / grid_count;
                uint32_t b = iz * 255 / grid_count;
                uint32_t color = pixel_rgba(r, g, b, 255);
                canvas_circle(cav, (x + 1) / 2 * WIDTH, (y + 1) / 2 * HEIGHT, circle_radius, color);
            }
        }
    }
    return cav;
}

static inline void rotate_point(float angle, float* x, float* y) {
    float dx = *x - WIDTH / 2;
    float dy = *y - HEIGHT / 2;
    float mag = sqrtf(dx * dx + dy * dy);
    float dir = atan2f(dy, dx) + angle;
    *x = cosf(dir) * mag + WIDTH / 2;
    *y = sinf(dir) * mag + HEIGHT / 2;
}

canvas_t* cat_cav = NULL;
canvas_t* triangle(canvas_t* cav, float dt) {
    // Parameters
    float angle = 0;
    float circle_radius = 100;
    uint32_t circle_color = 0x99AA2020;
    float circle_x = WIDTH/2;
    float circle_y = HEIGHT/2;
    float circle_dx = 100;
    float circle_dy = 100;

    // 0. init canvas
    fprintf(stderr, "cav: %p (%lu, %lu) %p\n", cav, cav->width, cav->height, cav->pixels);
    canvas_fill(cav, BG_COLOR);
    if(dt == 0.0) {
        
        image_t* cat = image_load("mobilenet_origin.jpg");
        image_resize(cat, 120, 120);
        cat_cav = canvas_from_image(cat);
    }

    // 1. Triangle
    {
        angle += 0.25f*PI*dt;

        float x1 = WIDTH/2, y1 = HEIGHT/8;
        float x2 = WIDTH/8, y2 = HEIGHT/2;
        float x3 = WIDTH*7/8, y3 = HEIGHT*7/8;
        rotate_point(angle, &x1, &y1);
        rotate_point(angle, &x2, &y2);
        rotate_point(angle, &x3, &y3);
        canvas_triangle_3c(cav, x1, y1, x2, y2, x3, y3, 0xFF2020FF, 0xFF20FF20, 0xFFFF2020);
    }

    // 2. Circle
    {
        float x = circle_x + circle_dx*dt;
        if (x - circle_radius < 0 || x + circle_radius >= WIDTH) {
            circle_dx *= -1;
        } else {
            circle_x = x;
        }

        float y = circle_y + circle_dy*dt;
        if (y - circle_radius < 0 || y + circle_radius >= HEIGHT) {
            circle_dy *= -1;
        } else {
            circle_y = y;
        }
        canvas_circle(cav, circle_x, circle_y, circle_radius, circle_color);
    }

    // 3. cat photo
    {
        canvas_draw(cav, 200 * 0.1 * dt, 300 * 0.8 * dt, cat_cav->width, cat_cav->height, cat_cav->pixels);
    }

    return cav;
}


UnitTest_fn_def(test_render_canvas) {
    renderer_t* rd = renderer_new(WIDTH, HEIGHT, RENDERER_TYPE_GIF);
    renderer_run(rd, triangle);
    renderer_free(rd);
    return NULL;
}

UnitTest_fn_def(test_render_linux) {
    renderer_t* rd = renderer_new(WIDTH, HEIGHT, RENDERER_TYPE_LINUX);
    renderer_run(rd, triangle);
    renderer_free(rd);
    return NULL;
}


UnitTest_fn_def(test_all) {
    UnitTest_add(test_render_linux);
    UnitTest_add(test_render_canvas);
    return NULL;
}

UnitTest_run(test_all);