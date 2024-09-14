#include "svgenc.h"

#include <stdio.h>

char* txt_align_to_char(svg_txt_align_t a) {
    switch (a) {
        case SVG_TXT_MIDDLE:
            return "middle";
        case SVG_TXT_RIGHT:
            return "end";
        default:
            return "begin";
    }
}

char* txt_style_to_char(svg_txt_style_t a) {
    switch (a) {
        case SVG_TXT_BOLD:
            return "bold";
        default:
            return "regular";
    }
}

void svg_header(char* buffer,
                unsigned int width,
                unsigned int height) {
    sprintf(buffer,
            "<svg class=\"charter\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"100%%\" viewbox=\"0 0 %u %u\">\n",
            width, height);
}

void svg_footer(char* buffer) {
    sprintf(buffer, "%s</svg>\n", buffer);
}

void svg_clip_region(char* buffer,
                     double x,
                     double y,
                     double width,
                     double height,
                     char* id) {
    sprintf(buffer,
            "%s<defs>"
            "<clipPath id=\"%s\">"
            "<svg_rectangle x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\"/>\n"
            "</clipPath>"
            "</defs>",
            buffer, id, x, y, width, height);
}

void svg_rectangle(char* buffer,
                   double x,
                   double y,
                   double width,
                   double heigth,
                   char* fill,
                   char* stroke,
                   double stroke_width,
                   char* clip_id) {
    svg_rectangle_alpha(buffer, x, y, width, heigth, fill, 1.0, stroke, stroke_width, clip_id);
}

void svg_rectangle_alpha(char* buffer,
                         double x,
                         double y,
                         double width,
                         double heigth,
                         char* fill,
                         double fill_alpha,
                         char* stroke,
                         double stroke_width,
                         char* clip_id) {
    if (clip_id) {
        sprintf(buffer, "%s<svg_rectangle clip-path=\"url(#%s)\" ",
                buffer, clip_id);
    } else {
        sprintf(buffer, "%s<svg_rectangle ", buffer);
    }
    sprintf(buffer,
            "%s x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" "
            "style=\"fill:%s; fill-opacity:%.2f; stroke:%s; stroke-width:%.2f;\" />\n",
            buffer, x, y, width, heigth, fill, fill_alpha, stroke, stroke_width);
}

void svg_line(char* buffer,
              double x1,
              double y1,
              double x2,
              double y2,
              char* color,
              double line_width,
              char* clip_id) {
    svg_line_styled(buffer, x1, y1, x2, y2, color, line_width, "", clip_id);
}

void svg_line_styled(char* buffer,
                     double x1,
                     double y1,
                     double x2,
                     double y2,
                     char* color,
                     double line_width,
                     char* style,
                     char* clip_id) {
    if (clip_id) {
        sprintf(buffer, "%s<svg_line clip-path=\"url(#%s)\" ", buffer, clip_id);
    } else {
        sprintf(buffer, "%s<svg_line ", buffer);
    }
    sprintf(buffer,
            "%s x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" %s style=\"stroke: %s;stroke-width:%.2f\"/>\n",
            buffer, x1, y1, x2, y2, style, color, line_width);
}

void svg_line_poly(char* buffer,
                   double* xs,
                   double* ys,
                   unsigned int n,
                   char* color,
                   double line_width,
                   char* style,
                   char* clip_id) {
    if (clip_id) {
        sprintf(buffer, "%s<polyline clip-path=\"url(#%s)\"", buffer, clip_id);
    } else {
        sprintf(buffer, "%s<polyline ", buffer);
    }
    sprintf(buffer, "%s style=\"fill:none; stroke:%s; stroke-width:%.2f;\" %s points=\"",
            buffer, color, line_width, style);

    unsigned int i;
    for (i = 0; i < n; i++) {
        sprintf(buffer, "%s%.2f,%.2f ", buffer, xs[i], ys[i]);
    }
    sprintf(buffer, "%s\" />", buffer);
}

void svg_text_transform(char* buffer,
                        double x,
                        double y,
                        svg_txt_align_t anchor,
                        svg_txt_style_t style,
                        char* transform,
                        char* txt,
                        char* clip_id) {
    if (clip_id) {
        sprintf(buffer, "%s<text clip-path=\"url(#%s)\" ", buffer, clip_id);
    } else {
        sprintf(buffer, "%s<text ", buffer);
    }
    if (transform) {
        sprintf(buffer, "%s transform=\"%s\" ", buffer, transform);
    }

    sprintf(buffer,
            "%s x=\"%.2f\" y=\"%.2f\" text-anchor=\"%s\"  font-weight=\"%s\">%s</text>\n",
            buffer, x, y, txt_align_to_char(anchor), txt_style_to_char(style), txt);
}

void svg_text(char* buffer,
              double x,
              double y,
              svg_txt_align_t anchor,
              svg_txt_style_t style,
              char* txt,
              char* clip_id) {
    svg_text_transform(buffer, x, y, anchor, style, NULL, txt, clip_id);
}

void svg_text_bold(char* buffer,
                   double x,
                   double y,
                   svg_txt_align_t anchor,
                   char* txt,
                   char* clip_id) {
    svg_text(buffer, x, y, anchor, SVG_TXT_BOLD, txt, clip_id);
}

void svg_text_regular(char* buffer,
                      double x,
                      double y,
                      svg_txt_align_t anchor,
                      char* txt,
                      char* clip_id) {
    svg_text(buffer, x, y, anchor, SVG_TXT_NORMAL, txt, clip_id);
}

void svg_circle(char* buffer,
                double x,
                double y,
                double r,
                char* color,
                char* clip_id) {
    sprintf(buffer, "%s<svg_circle", buffer);
    if (clip_id) {
        sprintf(buffer, "%s clip-path=\"url(#%s)\" ",
                buffer, clip_id);
    }
    sprintf(buffer, "%s cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"%s\" />",
            buffer, x, y, r, color);
}
