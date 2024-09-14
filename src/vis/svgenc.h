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
                     double x,
                     double y,
                     double width,
                     double height,
                     char* id);

void svg_rectangle(char* buffer,
                   double x,
                   double y,
                   double width,
                   double heigth,
                   char* fill,
                   char* stroke,
                   double stroke_width,
                   char* clip_id);

void svg_rectangle_alpha(char* buffer,
                         double x,
                         double y,
                         double width,
                         double heigth,
                         char* fill,
                         double fill_alpha,
                         char* stroke,
                         double stroke_width,
                         char* clip_id);

void svg_line_styled(char* buffer,
                     double x1,
                     double y1,
                     double x2,
                     double y2,
                     char* color,
                     double line_width,
                     char* style,
                     char* clip_id);

void svg_line(char* buffer,
              double x1,
              double y1,
              double x2,
              double y2,
              char* color,
              double line_width,
              char* clip_id);

void svg_line_poly(char* buffer,
                   double* xs,
                   double* ys,
                   unsigned int n,
                   char* color,
                   double line_width,
                   char* style,
                   char* clip_id);

void svg_text_transform(char* buffer,
                        double x,
                        double y,
                        svg_txt_align_t anchor,
                        svg_txt_style_t style,
                        char* transform,
                        char* text,
                        char* clip_id);

void svg_text(char* buffer,
              double x,
              double y,
              svg_txt_align_t anchor,
              svg_txt_style_t style,
              char* text,
              char* clip_id);

void svg_text_bold(char* buffer,
                   double x,
                   double y,
                   svg_txt_align_t anchor,
                   char* text,
                   char* clip_id);

void svg_text_regular(char* buffer,
                      double x,
                      double y,
                      svg_txt_align_t anchor,
                      char* text,
                      char* clip_id);

void svg_circle(char* buffer,
                double x,
                double y,
                double r,
                char* color,
                char* clip_id);

#endif  // SVGENC_H