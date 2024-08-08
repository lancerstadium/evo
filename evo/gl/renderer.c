#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../evo.h"
#include "../util/log.h"
#include "../util/math.h"
#include "../util/sys.h"

// ==================================================================================== //
//                                  renderer: term
// ==================================================================================== //

#define TERM_SCALE_DOWN_FACTOR 20

typedef struct {
    size_t actual_width;
    size_t actual_height;
    size_t scalaed_down_width;
    size_t scalaed_down_height;
    int* char_canvas;
} renderer_term_t;

int hsl256[][3] = {
    {0, 0, 0},
    {0, 100, 25},
    {120, 100, 25},
    {60, 100, 25},
    {240, 100, 25},
    {300, 100, 25},
    {180, 100, 25},
    {0, 0, 75},
    {0, 0, 50},
    {0, 100, 50},
    {120, 100, 50},
    {60, 100, 50},
    {240, 100, 50},
    {300, 100, 50},
    {180, 100, 50},
    {0, 0, 100},
    {0, 0, 0},
    {240, 99, 18},
    {240, 100, 26},
    {240, 100, 34},
    {240, 100, 42},
    {240, 100, 50},
    {120, 99, 18},
    {180, 99, 18},
    {197, 100, 26},
    {207, 100, 34},
    {213, 100, 42},
    {217, 100, 50},
    {120, 100, 26},
    {162, 100, 26},
    {180, 100, 26},
    {193, 100, 34},
    {202, 100, 42},
    {208, 100, 50},
    {120, 100, 34},
    {152, 100, 34},
    {166, 100, 34},
    {180, 100, 34},
    {191, 100, 42},
    {198, 100, 50},
    {120, 100, 42},
    {146, 100, 42},
    {157, 100, 42},
    {168, 100, 42},
    {180, 100, 42},
    {189, 100, 50},
    {120, 100, 50},
    {142, 100, 50},
    {151, 100, 50},
    {161, 100, 50},
    {170, 100, 50},
    {180, 100, 50},
    {0, 99, 18},
    {300, 99, 18},
    {282, 100, 26},
    {272, 100, 34},
    {266, 100, 42},
    {262, 100, 50},
    {60, 99, 18},
    {0, 0, 37},
    {240, 17, 45},
    {240, 33, 52},
    {240, 60, 60},
    {240, 100, 68},
    {77, 100, 26},
    {120, 17, 45},
    {180, 17, 45},
    {210, 33, 52},
    {220, 60, 60},
    {225, 100, 68},
    {87, 100, 34},
    {120, 33, 52},
    {150, 33, 52},
    {180, 33, 52},
    {200, 60, 60},
    {210, 100, 68},
    {93, 100, 42},
    {120, 60, 60},
    {140, 60, 60},
    {160, 60, 60},
    {180, 60, 60},
    {195, 100, 68},
    {97, 100, 50},
    {120, 100, 68},
    {135, 100, 68},
    {150, 100, 68},
    {165, 100, 68},
    {180, 100, 68},
    {0, 100, 26},
    {317, 100, 26},
    {300, 100, 26},
    {286, 100, 34},
    {277, 100, 42},
    {271, 100, 50},
    {42, 100, 26},
    {0, 17, 45},
    {300, 17, 45},
    {270, 33, 52},
    {260, 60, 60},
    {255, 100, 68},
    {60, 100, 26},
    {60, 17, 45},
    {0, 0, 52},
    {240, 20, 60},
    {240, 50, 68},
    {240, 100, 76},
    {73, 100, 34},
    {90, 33, 52},
    {120, 20, 60},
    {180, 20, 60},
    {210, 50, 68},
    {220, 100, 76},
    {82, 100, 42},
    {100, 60, 60},
    {120, 50, 68},
    {150, 50, 68},
    {180, 50, 68},
    {200, 100, 76},
    {88, 100, 50},
    {105, 100, 68},
    {120, 100, 76},
    {140, 100, 76},
    {160, 100, 76},
    {180, 100, 76},
    {0, 100, 34},
    {327, 100, 34},
    {313, 100, 34},
    {300, 100, 34},
    {288, 100, 42},
    {281, 100, 50},
    {32, 100, 34},
    {0, 33, 52},
    {330, 33, 52},
    {300, 33, 52},
    {280, 60, 60},
    {270, 100, 68},
    {46, 100, 34},
    {30, 33, 52},
    {0, 20, 60},
    {300, 20, 60},
    {270, 50, 68},
    {260, 100, 76},
    {60, 100, 34},
    {60, 33, 52},
    {60, 20, 60},
    {0, 0, 68},
    {240, 33, 76},
    {240, 100, 84},
    {71, 100, 42},
    {80, 60, 60},
    {90, 50, 68},
    {120, 33, 76},
    {180, 33, 76},
    {210, 100, 84},
    {78, 100, 50},
    {90, 100, 68},
    {100, 100, 76},
    {120, 100, 84},
    {150, 100, 84},
    {180, 100, 84},
    {0, 100, 42},
    {333, 100, 42},
    {322, 100, 42},
    {311, 100, 42},
    {300, 100, 42},
    {290, 100, 50},
    {26, 100, 42},
    {0, 60, 60},
    {340, 60, 60},
    {320, 60, 60},
    {300, 60, 60},
    {285, 100, 68},
    {37, 100, 42},
    {20, 60, 60},
    {0, 50, 68},
    {330, 50, 68},
    {300, 50, 68},
    {280, 100, 76},
    {48, 100, 42},
    {40, 60, 60},
    {30, 50, 68},
    {0, 33, 76},
    {300, 33, 76},
    {270, 100, 84},
    {60, 100, 42},
    {60, 60, 60},
    {60, 50, 68},
    {60, 33, 76},
    {0, 0, 84},
    {240, 100, 92},
    {69, 100, 50},
    {75, 100, 68},
    {80, 100, 76},
    {90, 100, 84},
    {120, 100, 92},
    {180, 100, 92},
    {0, 100, 50},
    {337, 100, 50},
    {328, 100, 50},
    {318, 100, 50},
    {309, 100, 50},
    {300, 100, 50},
    {22, 100, 50},
    {0, 100, 68},
    {345, 100, 68},
    {330, 100, 68},
    {315, 100, 68},
    {300, 100, 68},
    {31, 100, 50},
    {15, 100, 68},
    {0, 100, 76},
    {340, 100, 76},
    {320, 100, 76},
    {300, 100, 76},
    {41, 100, 50},
    {30, 100, 68},
    {20, 100, 76},
    {0, 100, 84},
    {330, 100, 84},
    {300, 100, 84},
    {50, 100, 50},
    {45, 100, 68},
    {40, 100, 76},
    {30, 100, 84},
    {0, 100, 92},
    {300, 100, 92},
    {60, 100, 50},
    {60, 100, 68},
    {60, 100, 76},
    {60, 100, 84},
    {60, 100, 92},
    {0, 0, 100},
    {0, 0, 3},
    {0, 0, 7},
    {0, 0, 10},
    {0, 0, 14},
    {0, 0, 18},
    {0, 0, 22},
    {0, 0, 26},
    {0, 0, 30},
    {0, 0, 34},
    {0, 0, 38},
    {0, 0, 42},
    {0, 0, 46},
    {0, 0, 50},
    {0, 0, 54},
    {0, 0, 58},
    {0, 0, 61},
    {0, 0, 65},
    {0, 0, 69},
    {0, 0, 73},
    {0, 0, 77},
    {0, 0, 81},
    {0, 0, 85},
    {0, 0, 89},
    {0, 0, 93},
};

int distance_hsl256(int i, int h, int s, int l) {
    int dh = h - hsl256[i][0];
    int ds = s - hsl256[i][1];
    int dl = l - hsl256[i][2];
    return dh * dh + ds * ds + dl * dl;
}

// TODO: bring find_ansi_index_by_rgb from image2term
int find_ansi_index_by_hsl(int h, int s, int l) {
    int index = 0;
    for (int i = 0; i < 256; ++i) {
        if (distance_hsl256(i, h, s, l) < distance_hsl256(index, h, s, l)) {
            index = i;
        }
    }
    return index;
}

void rgb_to_hsl(int r, int g, int b, int* h, int* s, int* l) {
    float r01 = r / 255.0f;
    float g01 = g / 255.0f;
    float b01 = b / 255.0f;
    float cmax = r01;
    if (g01 > cmax) cmax = g01;
    if (b01 > cmax) cmax = b01;
    float cmin = r01;
    if (g01 < cmin) cmin = g01;
    if (b01 < cmin) cmin = b01;
    float delta = cmax - cmin;
    float epsilon = 1e-6;
    float hf = 0;
    if (delta < epsilon)
        hf = 0;
    else if (cmax == r01)
        hf = 60.0f * fmod((g01 - b01) / delta, 6.0f);
    else if (cmax == g01)
        hf = 60.0f * ((b01 - r01) / delta + 2);
    else if (cmax == b01)
        hf = 60.0f * ((r01 - g01) / delta + 4);
    else {
        LOG_ERR("rgb_to_hsl: cmax is not r01, g01 or b01\n");
        return;
    }

    float lf = (cmax + cmin) / 2;

    float sf = 0;
    if (delta < epsilon)
        sf = 0;
    else
        sf = delta / (1 - fabsf(2 * lf - 1));

    *h = fmodf(fmodf(hf, 360.0f) + 360.0f, 360.0f);
    *s = sf * 100.0f;
    *l = lf * 100.0f;
}

static uint32_t renderer_term_compress_pixel_chunk(canvas_t* cav) {
    size_t r = 0;
    size_t g = 0;
    size_t b = 0;
    size_t a = 0;

    for (size_t y = 0; y < cav->height; ++y) {
        for (size_t x = 0; x < cav->width; ++x) {
            r += pixel_red(canvas_pixel(cav, x, y));
            g += pixel_green(canvas_pixel(cav, x, y));
            b += pixel_blue(canvas_pixel(cav, x, y));
            a += pixel_alpha(canvas_pixel(cav, x, y));
        }
    }

    r /= cav->width * cav->height;
    g /= cav->width * cav->height;
    b /= cav->width * cav->height;
    a /= cav->width * cav->height;

    return pixel_rgba(r, g, b, a);
}

static void renderer_term_resize_char_canvas(renderer_t* rd, size_t width, size_t height) {
    if (!rd || !rd->priv) return;
    renderer_term_t* priv = rd->priv;
    if (width % TERM_SCALE_DOWN_FACTOR != 0 || height % TERM_SCALE_DOWN_FACTOR != 0) {
        LOG_WARN("renderer term: width and height must be divisible by %d\n", TERM_SCALE_DOWN_FACTOR);
        return;
    }
    priv->actual_width = width;
    priv->actual_height = height;
    priv->scalaed_down_width = width / TERM_SCALE_DOWN_FACTOR;
    priv->scalaed_down_height = height / TERM_SCALE_DOWN_FACTOR;
    if (priv->char_canvas) { 
        free(priv->char_canvas);
        priv->char_canvas = NULL;
    }
    priv->char_canvas = malloc(sizeof(*priv->char_canvas) * priv->scalaed_down_width * priv->scalaed_down_height);
    if (priv->char_canvas == NULL) {
        LOG_WARN("renderer term: failed to allocate char canvas\n");
    }
}

static void renderer_term_compress_pixel(renderer_t* rd, canvas_t* cav) {
    if (!rd || !rd->priv || !cav) return;
    renderer_term_t* priv = rd->priv;
    if (priv->actual_width != cav->width || priv->actual_height != cav->height) {
        renderer_term_resize_char_canvas(rd, cav->width, cav->height);
    }
    for (size_t y = 0; y < priv->scalaed_down_height; y++) {
        for (size_t x = 0; x < priv->scalaed_down_width; x++) {
            canvas_t* sub_cav = canvas_sub_new(cav, x * TERM_SCALE_DOWN_FACTOR, y * TERM_SCALE_DOWN_FACTOR, TERM_SCALE_DOWN_FACTOR, TERM_SCALE_DOWN_FACTOR);
            uint32_t chunk_pixel = renderer_term_compress_pixel_chunk(sub_cav);
            int r = pixel_red(chunk_pixel);
            int g = pixel_green(chunk_pixel);
            int b = pixel_blue(chunk_pixel);
            int a = pixel_alpha(chunk_pixel);
            r = a * (r / 255.0f);
            g = a * (g / 255.0f);
            b = a * (b / 255.0f);
            int h, s, l;
            rgb_to_hsl(r, g, b, &h, &s, &l);
            priv->char_canvas[y * priv->scalaed_down_width + x] = find_ansi_index_by_hsl(h, s, l);
            // canvas_free(sub_cav);
        }
    }
}

void renderer_render_term(renderer_t* rd, render_fn_t rd_fn) {
    if (!rd || !rd->priv || !rd_fn) return;
    renderer_term_t* priv = rd->priv;
    canvas_t* cav = NULL;
    for (;;) {
        cav = rd_fn(1.f / 60.f);
        renderer_term_compress_pixel(rd, cav);
        for (size_t y = 0; y < priv->actual_height; y++) {
            for (size_t x = 0; x < priv->actual_width; x++) {
                printf("\033[48;5;%dm  ", priv->char_canvas[y * priv->actual_width + x]);
            }
            printf("\033[0m\n");
        }
        usleep(1000 * 1000 / 60);
        printf("\033[%zuA", priv->actual_height);
        printf("\033[%zuD", priv->actual_width);
    }
    canvas_free(cav);
}

// ==================================================================================== //
//                                  renderer: gif
// ==================================================================================== //

typedef struct {
    
} renderer_gif_t;

void renderer_render_gif(renderer_t* rd, render_fn_t rd_fn) {
    if (!rd || !rd->priv || !rd_fn) return;
    renderer_gif_t* priv = rd->priv;
    canvas_t* cav = rd_fn(0);
    int64_t ndelay = 60;        // 0 < ndelay <= 60
    int64_t delays[ndelay];
    int64_t delay = 5;          // 5 ms
    delays[0] = delay;
    for (size_t i = 1; i < ndelay; i++) {
        delays[i] = delay;
        canvas_t* cav_tmp = rd_fn((1.f * i) / 60.f);
        image_push(cav->background, cav_tmp->background);
    }
    image_set_deloys(cav->background, delays, ndelay);
    canvas_export(cav, "renderer.gif");
    canvas_free(cav);
}

// ==================================================================================== //
//                                  renderer: API
// ==================================================================================== //

renderer_t* renderer_new(renderer_type_t type) {
    renderer_t* rd = malloc(sizeof(renderer_t));
    rd->type = type;
    switch (type) {
        case RENDERER_TYPE_TERM: {
            renderer_term_t* priv = malloc(sizeof(renderer_term_t));
            if (priv) {
                priv->actual_height = 0;
                priv->actual_width = 0;
                priv->scalaed_down_height = 0;
                priv->scalaed_down_width = 0;
                priv->char_canvas = NULL;
            }
            rd->priv = priv;
            rd->render = renderer_render_term;
            break;
        }
        case RENDERER_TYPE_GIF: {
            renderer_gif_t* priv = malloc(sizeof(renderer_gif_t));
            if (priv) {
                
            }
            rd->priv = priv;
            rd->render = renderer_render_gif;
            break;
        }
        default:
            rd->priv = NULL;
            rd->render = NULL;
            break;
    }
    return rd;
}

void renderer_run(renderer_t* rd, render_fn_t rd_fn) {
    if (rd && rd->render && rd_fn) {
        rd->render(rd, rd_fn);
    }
}

void renderer_free(renderer_t* rd) {
    if (rd) {
        if (rd->priv) {
            free(rd->priv);
        }
        free(rd);
        rd = NULL;
    }
}
