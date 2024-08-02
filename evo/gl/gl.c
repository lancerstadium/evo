#include <stdio.h>
#include <string.h>

#include "../evo.h"
#include "../util/log.h"
#include "../util/math.h"
#include "../util/sys.h"

#define pixel_red(color) (((color) & 0x000000FF) >> (8 * 0))
#define pixel_green(color) (((color) & 0x0000FF00) >> (8 * 1))
#define pixel_blue(color) (((color) & 0x00FF0000) >> (8 * 2))
#define pixel_alpha(color) (((color) & 0xFF000000) >> (8 * 3))
#define pixel_rgba(r, g, b, a) ((((r) & 0xFF) << (8 * 0)) | (((g) & 0xFF) << (8 * 1)) | (((b) & 0xFF) << (8 * 2)) | (((a) & 0xFF) << (8 * 3)))

#define CANVAS_AA_RES 2
#define CANVAS_SWAP(T, a, b) \
    do {                     \
        T t = a;             \
        a = b;               \
        b = t;               \
    } while (0)
#define CANVAS_SIGN(T, x) ((T)((x) > 0) - (T)((x) < 0))
#define CANVAS_ABS(T, x) (CANVAS_SIGN(T, x) * (x))

canvas_t* canvas_from_image(image_t* img) {
    if (!img || !img->raw) return NULL;
    canvas_t* cav = NULL;
    tensor_t* temp = NULL;
    if(img->raw->layout == 0) {
        cav = canvas_new(img->raw->dims[3], img->raw->dims[2]);
        temp = tensor_nchw2nhwc(img->raw);
    } else {
        cav = canvas_new(img->raw->dims[2], img->raw->dims[1]);
        temp = img->raw;
    }
    if (cav) {
        uint8_t* data = (uint8_t*)temp->datas;
        for (int h = 0; h < temp->dims[1]; ++h) {
            for (int w = 0; w < temp->dims[2]; ++w) {
                uint32_t color = 0;
                for (int c = 0; c < temp->dims[3]; ++c) {
                    int idx = c + (h * temp->dims[2] + w) * temp->dims[3];
                    color |= data[idx] << (8 * c);
                }
                canvas_pixel(cav, w, h) = color;
            }
        }
        if(img->raw->layout == 0) {
            tensor_free(temp);
        }
        return cav;
    }
    return NULL;
}

canvas_t* canvas_new(size_t width, size_t height) {
    canvas_t* cav = sys_malloc(sizeof(canvas_t));
    if (cav) {
        cav->background = image_blank("bg", MAX(width, 1), MAX(height, 1));
        return cav;
    }
    return NULL;
}

uint32_t* canvas_get(canvas_t* cav, size_t x, size_t y) {
    if (cav && cav->background && cav->background->raw && cav->background->raw->datas) {
        uint32_t* data = (uint32_t*)((cav)->background->raw->datas);
        return &data[(x) + (canvas_width(cav)) * (y)];
    }
    return NULL;
}

void canvas_fill(canvas_t* cav, uint32_t color) {
    if (!cav) return;
    for (size_t y = 0; y < canvas_height(cav); y++) {
        for (size_t x = 0; x < canvas_width(cav); x++) {
            canvas_pixel(cav, x, y) = color;
        }
    }
}

void canvas_blend(uint32_t* pixel, uint32_t color) {
    uint32_t r1 = pixel_red(*pixel);
    uint32_t g1 = pixel_green(*pixel);
    uint32_t b1 = pixel_blue(*pixel);
    uint32_t a1 = pixel_alpha(*pixel);

    uint32_t r2 = pixel_red(color);
    uint32_t g2 = pixel_green(color);
    uint32_t b2 = pixel_blue(color);
    uint32_t a2 = pixel_alpha(color);

    r1 = (r1 * (255 - a2) + r2 * a2) / 255;
    if (r1 > 255) r1 = 255;
    g1 = (g1 * (255 - a2) + g2 * a2) / 255;
    if (g1 > 255) g1 = 255;
    b1 = (b1 * (255 - a2) + b2 * a2) / 255;
    if (b1 > 255) b1 = 255;

    *pixel = pixel_rgba(r1, g1, b1, a1);
}

uint32_t color_interpolate(uint32_t color1, uint32_t color2, float t) {
    if(t > 1.0f) t = 1.0f;
    if(t < 0.0f) t = 0.0f;
    uint8_t r1 = pixel_red(color1);
    uint8_t g1 = pixel_green(color1);
    uint8_t b1 = pixel_blue(color1);
    uint8_t a1 = pixel_alpha(color1);

    uint8_t r2 = pixel_red(color2);
    uint8_t g2 = pixel_green(color2);
    uint8_t b2 = pixel_blue(color2);
    uint8_t a2 = pixel_alpha(color2);

    uint8_t r = r1 + t * (r2 - r1);
    uint8_t g = g1 + t * (g2 - g1);
    uint8_t b = b1 + t * (b2 - b1);
    uint8_t a = a1 + t * (a2 - a1);

    return pixel_rgba(r, g, b, a);
}

bool canvas_is_in_bound(canvas_t* cav, int x, int y) {
    if (!cav) return false;
    return 0 <= x && x < canvas_width(cav) && 0 <= y && y < canvas_height(cav);
}

void canvas_line(canvas_t* cav, int x1, int y1, int x2, int y2, uint32_t color) {
    if (!cav) return;
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
            int y = dy * (x - x1) / dx + y1;
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
            int x = dx * (y - y1) / dy + x1;
            // TODO: move boundary checks out side of the loops in olivec_draw_line
            if (canvas_is_in_bound(cav, x, y)) {
                canvas_blend(&canvas_pixel(cav, x, y), color);
            }
        }
    }
}

bool canvas_normalize_rectangle(canvas_t* cav, int x, int y, int w, int h, rectangle_t* rec) {
    if (w == 0 || h == 0) return false;
    int ox1 = x;
    int oy1 = y;
    int ox2 = ox1 + CANVAS_SIGN(int, w) * (CANVAS_ABS(int, w) - 1);
    if (ox1 > ox2) CANVAS_SWAP(int, ox1, ox2);
    int oy2 = oy1 + CANVAS_SIGN(int, h) * (CANVAS_ABS(int, h) - 1);
    if (oy1 > oy2) CANVAS_SWAP(int, oy1, oy2);

    if (ox1 >= canvas_width(cav) || ox2 < 0) return false;
    if (oy1 >= canvas_height(cav) || oy2 < 0) return false;

    rec->x1 = ox1;
    rec->x2 = ox2;
    rec->y1 = oy1;
    rec->y2 = oy2;

    if (rec->x1 < 0) rec->x1 = 0;
    if (rec->x2 >= canvas_width(cav)) rec->x2 = canvas_width(cav) - 1;
    if (rec->y1 < 0) rec->y1 = 0;
    if (rec->y2 >= canvas_height(cav)) rec->y2 = canvas_height(cav) - 1;
    return true;
}

void canvas_rectangle(canvas_t* cav, int x, int y, int w, int h, uint32_t color) {
    if (!cav) return;
    for (int i = x; i != MIN(x + w, canvas_width(cav) - 1) && i >= 0; i += CANVAS_SIGN(int, w)) {
        for (int j = y; j != MIN(y + h, canvas_height(cav) - 1) && j >= 0; j += CANVAS_SIGN(int, h)) {
            canvas_blend(&canvas_pixel(cav, i, j), color);
        }
    }
}

void canvas_rectangle_c2(canvas_t* cav, int x, int y, int w, int h, uint32_t color1, uint32_t color2) {
    if (!cav) return;
    for (int i = x; i != MIN(x + w, canvas_width(cav) - 1) && i >= 0; i += CANVAS_SIGN(int, w)) {
        float t_x = (float)(i - x) / w;
        for (int j = y; j != MIN(y + h, canvas_height(cav) - 1) && j >= 0; j += CANVAS_SIGN(int, h)) {
            float t_y = (float)(j - y) / h;
            uint32_t color = color_interpolate(color1, color2, (t_x + t_y) / 2.0f);
            canvas_blend(&canvas_pixel(cav, i, j), color);
        }
    }
}


void canvas_frame(canvas_t* cav, int x, int y, int w, int h, size_t t, uint32_t color) {
    if (!cav || t == 0) return;
    int x1 = x;
    int y1 = y;
    int x2 = x1 + CANVAS_SIGN(int, w) * (CANVAS_ABS(int, w) - 1);
    if (x1 > x2) CANVAS_SWAP(int, x1, x2);
    int y2 = y1 + CANVAS_SIGN(int, h) * (CANVAS_ABS(int, h) - 1);
    if (y1 > y2) CANVAS_SWAP(int, y1, y2);

    canvas_rectangle(cav, x1 - t / 2, y1 - t / 2, (x2 - x1 + 1) + t / 2 * 2, t, color);   // Top
    canvas_rectangle(cav, x1 - t / 2, y1 - t / 2, t, (y2 - y1 + 1) + t / 2 * 2, color);   // Left
    canvas_rectangle(cav, x1 - t / 2, y2 + t / 2, (x2 - x1 + 1) + t / 2 * 2, -t, color);  // Bottom
    canvas_rectangle(cav, x2 + t / 2, y1 - t / 2, -t, (y2 - y1 + 1) + t / 2 * 2, color);  // Right
}

void canvas_ellipse(canvas_t* cav, int cx, int cy, int rx, int ry, uint32_t color) {
    if (!cav) return;
    int rx1 = rx + CANVAS_SIGN(int, rx);
    int ry1 = ry + CANVAS_SIGN(int, ry);

    rectangle_t rec = {0};
    if (!canvas_normalize_rectangle(cav, cx - rx1, cy - ry1, 2 * rx1, 2 * ry1, &rec)) return;

    for (int y = rec.y1; y <= rec.y2; y++) {
        for (int x = rec.x1; x <= rec.x2; x++) {
            float nx = (x + 0.5 - rec.x1) / (2.0f * rx1);
            float ny = (y + 0.5 - rec.y1) / (2.0f * ry1);
            float dx = nx - 0.5;
            float dy = ny - 0.5;
            if (dx * dx + dy * dy <= 0.5 * 0.5) {
                canvas_pixel(cav, x, y) = color;
            }
        }
    }
}

void canvas_circle(canvas_t* cav, int cx, int cy, int r, uint32_t color) {
    if (!cav) return;
    rectangle_t rec = {0};
    int r1 = r + CANVAS_SIGN(int, r);
    if (!canvas_normalize_rectangle(cav, cx - r1, cy - r1, 2 * r1, 2 * r1, &rec)) return;

    for (int y = rec.y1; y <= rec.y2; ++y) {
        for (int x = rec.x1; x <= rec.x2; ++x) {
            int count = 0;
            for (int sox = 0; sox < CANVAS_AA_RES; ++sox) {
                for (int soy = 0; soy < CANVAS_AA_RES; ++soy) {
                    // TODO: switch to 64 bits to make the overflow less likely
                    // Also research the probability of overflow
                    int res1 = (CANVAS_AA_RES + 1);
                    int dx = (x * res1 * 2 + 2 + sox * 2 - res1 * cx * 2 - res1);
                    int dy = (y * res1 * 2 + 2 + soy * 2 - res1 * cy * 2 - res1);
                    if (dx * dx + dy * dy <= res1 * res1 * r * r * 2 * 2) count += 1;
                }
            }
            uint32_t alpha = ((color & 0xFF000000) >> (3 * 8)) * count / CANVAS_AA_RES / CANVAS_AA_RES;
            uint32_t updated_color = (color & 0x00FFFFFF) | (alpha << (3 * 8));
            canvas_blend(&canvas_pixel(cav, x, y), updated_color);
        }
    }
}

void canvas_text(canvas_t* cav, const char* text, int tx, int ty, font_t* font, size_t glyph_size, uint32_t color) {
    if (!cav || !text || !font) return;
    for (size_t i = 0; *text; ++i, ++text) {
        int gx = tx + i * font->width * glyph_size;
        int gy = ty;
        const char* glyph = &font->glyphs[(*text) * sizeof(char) * font->width * font->height];
        for (int dy = 0; (size_t)dy < font->height; ++dy) {
            for (int dx = 0; (size_t)dx < font->width; ++dx) {
                int px = gx + dx * glyph_size;
                int py = gy + dy * glyph_size;
                if (0 <= px && px < (int)canvas_width(cav) && 0 <= py && py < (int)canvas_height(cav)) {
                    if (glyph[dy * font->width + dx]) {
                        canvas_rectangle(cav, px, py, glyph_size, glyph_size, color);
                    }
                }
            }
        }
    }
}

void canvas_export(canvas_t* cav, const char* path) {
    if (cav && cav->background) {
        image_save(cav->background, path);
    }
}

void canvas_free(canvas_t* cav) {
    if (cav) {
        if (cav->background) image_free(cav->background);
        free(cav);
        cav = NULL;
    }
}