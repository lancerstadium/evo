#include "svgenc.h"

#include <math.h>
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
                     float x,
                     float y,
                     float width,
                     float height,
                     char* id) {
    sprintf(buffer,
            "%s<defs>"
            "<clipPath id=\"%s\">"
            "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\"/>\n"
            "</clipPath>"
            "</defs>",
            buffer, id, x, y, width, height);
}

void svg_rectangle(char* buffer,
                   float x,
                   float y,
                   float width,
                   float heigth,
                   char* fill,
                   char* stroke,
                   float stroke_width,
                   char* clip_id) {
    svg_rectangle_alpha(buffer, x, y, width, heigth, fill, 1.0, stroke, stroke_width, clip_id);
}

void svg_rectangle_alpha(char* buffer,
                         float x,
                         float y,
                         float width,
                         float heigth,
                         char* fill,
                         float fill_alpha,
                         char* stroke,
                         float stroke_width,
                         char* clip_id) {
    if (clip_id) {
        sprintf(buffer, "%s<rect clip-path=\"url(#%s)\" ",
                buffer, clip_id);
    } else {
        sprintf(buffer, "%s<rect ", buffer);
    }
    sprintf(buffer,
            "%s x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" "
            "style=\"fill:%s; fill-opacity:%.2f; stroke:%s; stroke-width:%.2f;\" />\n",
            buffer, x, y, width, heigth, fill, fill_alpha, stroke, stroke_width);
}

void svg_line(char* buffer,
              float x1,
              float y1,
              float x2,
              float y2,
              char* color,
              float line_width,
              char* clip_id) {
    svg_line_styled(buffer, x1, y1, x2, y2, color, line_width, "", clip_id);
}

void svg_line_styled(char* buffer,
                     float x1,
                     float y1,
                     float x2,
                     float y2,
                     char* color,
                     float line_width,
                     char* style,
                     char* clip_id) {
    if (clip_id) {
        sprintf(buffer, "%s<line clip-path=\"url(#%s)\" ", buffer, clip_id);
    } else {
        sprintf(buffer, "%s<line ", buffer);
    }
    sprintf(buffer,
            "%s x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" %s style=\"stroke: %s;stroke-width:%.2f\"/>\n",
            buffer, x1, y1, x2, y2, style, color, line_width);
}

void svg_line_poly(char* buffer,
                   float* xs,
                   float* ys,
                   unsigned int n,
                   char* color,
                   float line_width,
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
                        float x,
                        float y,
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
              float x,
              float y,
              svg_txt_align_t anchor,
              svg_txt_style_t style,
              char* txt,
              char* clip_id) {
    svg_text_transform(buffer, x, y, anchor, style, NULL, txt, clip_id);
}

void svg_text_bold(char* buffer,
                   float x,
                   float y,
                   svg_txt_align_t anchor,
                   char* txt,
                   char* clip_id) {
    svg_text(buffer, x, y, anchor, SVG_TXT_BOLD, txt, clip_id);
}

void svg_text_regular(char* buffer,
                      float x,
                      float y,
                      svg_txt_align_t anchor,
                      char* txt,
                      char* clip_id) {
    svg_text(buffer, x, y, anchor, SVG_TXT_NORMAL, txt, clip_id);
}

void svg_circle(char* buffer,
                float x,
                float y,
                float r,
                char* color,
                char* clip_id) {
    sprintf(buffer, "%s<circle", buffer);
    if (clip_id) {
        sprintf(buffer, "%s clip-path=\"url(#%s)\" ",
                buffer, clip_id);
    }
    sprintf(buffer, "%s cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"%s\" />",
            buffer, x, y, r, color);
}

void svg_bar(char* buffer,
              float lw,
              char* color,
              char* lcolor,
              float x,
              float y,
              float y0,
              float w,
              char* clip_id) {
    svg_rectangle(buffer, x - w / 2, (y < y0 ? y : y0), w, fabsf(y - y0), color, lcolor, lw, clip_id);
}

int svg_is_mark(char c) {
    switch (c)
    {
        case 'o':
            return 1;
        case 's':
            return 1;
        case 'x':
            return 1;
        case '+':
            return 1;
        default:
            return 0;
    }
}

void svg_point(char* buffer,
               char style,
               char* color,
               float x,
               float y,
               char* clip_id) {
    switch (style) {
        case 'o':
            svg_circle(buffer, x, y, 3, color, clip_id);
            break;
        case 's':
            svg_rectangle(buffer, x - 3, y - 3, 6, 6, color, "none", 0, clip_id);
            break;
        case 'x':
            svg_line(buffer, x - 3, y - 3, x + 3, y + 3, color, 1.5, clip_id);
            svg_line(buffer, x + 3, y - 3, x - 3, y + 3, color, 1.5, clip_id);
            break;
        case '+':
            svg_line(buffer, x - 4.5, y, x + 4.5, y, color, 1.5, clip_id);
            svg_line(buffer, x, y - 4.5, x, y + 4.5, color, 1.5, clip_id);
            break;
    }
}