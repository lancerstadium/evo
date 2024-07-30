#include <string.h>
#include <stdio.h>
#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#include "../util/math.h"


#define pixel_red(color)        (((color)&0x000000FF)>>(8*0))
#define pixel_green(color)      (((color)&0x0000FF00)>>(8*1))
#define pixel_blue(color)       (((color)&0x00FF0000)>>(8*2))
#define pixel_alpha(color)      (((color)&0xFF000000)>>(8*3))
#define pixel_rgba(r, g, b, a)  ((((r)&0xFF)<<(8*0)) | (((g)&0xFF)<<(8*1)) | (((b)&0xFF)<<(8*2)) | (((a)&0xFF)<<(8*3)))

#define CANVAS_SWAP(T, a, b) do { T t = a; a = b; b = t; } while (0)
#define CANVAS_SIGN(T, x)       ((T)((x) > 0) - (T)((x) < 0))
#define CANVAS_ABS(T, x)        (CANVAS_SIGN(T, x)*(x))

canvas_t* canvas_new(size_t height, size_t width) {
    canvas_t* cav = sys_malloc(sizeof(canvas_t));
    if(cav) {
        cav->background = image_blank("bg", MAX(height, 1), MAX(width, 1));
        return cav;
    }
    return NULL;
}

uint32_t* canvas_get(canvas_t* cav, size_t x, size_t y) {
    if(cav && cav->background && cav->background->raw && cav->background->raw->datas) {
        uint32_t* data = (uint32_t*)((cav)->background->raw->datas);
        return &data[(x) + (canvas_width(cav))*(y)];
    }
    return NULL;
}

void canvas_fill(canvas_t* cav, uint32_t color) {
    for(size_t y = 0; y < canvas_height(cav); y++) {
        for(size_t x = 0; x < canvas_width(cav); x++) {
            canvas_pixel(cav, x, y) = color;
        }
    }
}

void canvas_blend(uint32_t *pixel, uint32_t color) {
    uint32_t r1 = pixel_red(*pixel);
    uint32_t g1 = pixel_green(*pixel);
    uint32_t b1 = pixel_blue(*pixel);
    uint32_t a1 = pixel_alpha(*pixel);

    uint32_t r2 = pixel_red(color);
    uint32_t g2 = pixel_green(color);
    uint32_t b2 = pixel_blue(color);
    uint32_t a2 = pixel_alpha(color);

    r1 = (r1*(255 - a2) + r2*a2)/255; if (r1 > 255) r1 = 255;
    g1 = (g1*(255 - a2) + g2*a2)/255; if (g1 > 255) g1 = 255;
    b1 = (b1*(255 - a2) + b2*a2)/255; if (b1 > 255) b1 = 255;

    *pixel = pixel_rgba(r1, g1, b1, a1);
}

bool canvas_is_in_bound(canvas_t *cav, int x, int y) {
    return 0 <= x && x < canvas_width(cav) && 0 <= y && y < canvas_height(cav);
}

void canvas_line(canvas_t* cav, int x1, int y1, int x2, int y2, uint32_t color) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    if (dx == 0 && dy == 0) {
        if (canvas_is_in_bound(cav, x1, y1)) {
            canvas_blend(&canvas_pixel(cav, x1, y1), color);
        }
        return;
    }
    if (CANVAS_ABS(int, dx) > CANVAS_ABS(int, dy)) {
        if (x1 > x2) {
            CANVAS_SWAP(int, x1, x2);
            CANVAS_SWAP(int, y1, y2);
        }

        for (int x = x1; x <= x2; ++x) {
            int y = dy*(x - x1)/dx + y1;
            // TODO: move boundary checks out side of the loops in olivec_draw_line
            if (canvas_is_in_bound(cav, x, y)) {
                canvas_blend(&canvas_pixel(cav, x, y), color);
            }
        }
    } else {
        if (y1 > y2) {
            CANVAS_SWAP(int, x1, x2);
            CANVAS_SWAP(int, y1, y2);
        }

        for (int y = y1; y <= y2; ++y) {
            int x = dx*(y - y1)/dy + x1;
            // TODO: move boundary checks out side of the loops in olivec_draw_line
            if (canvas_is_in_bound(cav, x, y)) {
                canvas_blend(&canvas_pixel(cav, x, y), color);
            }
        }
    }
}

void canvas_export(canvas_t *cav, const char *path) {
    if(cav && cav->background) {
        image_save(cav->background, path);
    }
}

void canvas_free(canvas_t * cav) {
    if(cav) {
        if(cav->background) image_free(cav->background);
        free(cav);
        cav = NULL;
    }
}