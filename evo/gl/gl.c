#include <string.h>
#include <stdio.h>
#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#include "../util/math.h"

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
        return &data[(x) + ((cav)->background->raw->dims[3])*(y)];
    }
    return NULL;
}

void canvas_fill(canvas_t* cav, uint32_t color) {
    for(size_t y = 0; y < cav->background->raw->dims[2]; y++) {
        for(size_t x = 0; x < cav->background->raw->dims[3]; x++) {
            canvas_pixel(cav, x, y) = color;
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