#ifndef SVGENC_H

typedef enum {
    SVG_TXT_LEFT = 0,
    SVG_TXT_MIDDLE = 1,
    SVG_TXT_RIGHT = 2
} svg_txt_align_t;

typedef enum {
    SVG_TXT_NORMAL = 0,
    SVG_TXT_BOLD = 1
} svg_txt_style_t;

void svg_header(char* buffer,
                unsigned int width,
                unsigned int height);

void svg_footer(char* buffer);

void svg_clip_region(char* buffer,
                     float x,
                     float y,
                     float width,
                     float height,
                     char* id);

void svg_rectangle(char* buffer,
                   float x,
                   float y,
                   float width,
                   float heigth,
                   char* fill,
                   char* stroke,
                   float stroke_width,
                   char* clip_id);

void svg_rectangle_alpha(char* buffer,
                         float x,
                         float y,
                         float width,
                         float heigth,
                         char* fill,
                         float fill_alpha,
                         char* stroke,
                         float stroke_width,
                         char* clip_id);

void svg_line_styled(char* buffer,
                     float x1,
                     float y1,
                     float x2,
                     float y2,
                     char* color,
                     float line_width,
                     char* style,
                     char* clip_id);

void svg_line(char* buffer,
              float x1,
              float y1,
              float x2,
              float y2,
              char* color,
              float line_width,
              char* clip_id);

void svg_line_poly(char* buffer,
                   float* xs,
                   float* ys,
                   unsigned int n,
                   char* color,
                   float line_width,
                   char* style,
                   char* clip_id);

void svg_text_transform(char* buffer,
                        float x,
                        float y,
                        svg_txt_align_t anchor,
                        svg_txt_style_t style,
                        char* transform,
                        char* text,
                        char* clip_id);

void svg_text(char* buffer,
              float x,
              float y,
              svg_txt_align_t anchor,
              svg_txt_style_t style,
              char* text,
              char* clip_id);

void svg_text_bold(char* buffer,
                   float x,
                   float y,
                   svg_txt_align_t anchor,
                   char* text,
                   char* clip_id);

void svg_text_regular(char* buffer,
                      float x,
                      float y,
                      svg_txt_align_t anchor,
                      char* text,
                      char* clip_id);

void svg_circle(char* buffer,
                float x,
                float y,
                float r,
                char* color,
                char* clip_id);

int svg_is_mark(char c);

void svg_bar(char* buffer,
             float lw,
             char* color,
             char* lcolor,
             float x,
             float y,
             float y0,
             float w,
             char* clip_id);

void svg_point(char* buffer,
               char style,
               char* color,
               float x,
               float y,
               char* clip_id);

#endif  // SVGENC_H