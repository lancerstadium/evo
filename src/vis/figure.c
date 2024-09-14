#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/sys.h>
#include "svgenc.h"

#include <math.h>
#include <string.h>

// ==================================================================================== //
//                                  figure: color
// ==================================================================================== //

static const char * figure_colormap[10] = {
    "#db5f57",
    "#dbae57",
    "#b9db57",
    "#69db57",
    "#57db94",
    "#57d3db",
    "#5784db",
    "#7957db",
    "#c957db",
    "#db579e"
};

// ==================================================================================== //
//                                  figure: svg
// ==================================================================================== //

#define FIGURE_SVG_BUF_SIZE 1024 * 1024
#define FIGURE_SVG_X_MARGIN 35
#define FIGURE_SVG_Y_MARGIN 25

typedef struct {
    double x_min;
    double y_min;
    double x_max;
    double y_max;
    double x0;
    double y0;
    double top;
    double left;
    double w;
    double h;
} figure_priv_svg_t;


static void svg_title(char* buffer, char* title, double left, double width) {
    if(!buffer) return;
    double tx = left + width / 2;
    double ty = 15;
    svg_text_bold(buffer, tx, ty, SVG_TXT_MIDDLE, title ? title : "title", NULL);
}

static figure_priv_svg_t* figure_priv_svg(figure_t* fig) {
    if(!fig) return NULL;
    if(fig->priv) {
        free(fig->priv);
    }
    figure_priv_svg_t* priv = malloc(sizeof(figure_priv_svg_t));
    memset(priv, 0, sizeof(figure_priv_svg_t));
    // 1. Region
    int p_h = FIGURE_SVG_Y_MARGIN ;
    int p_w = FIGURE_SVG_X_MARGIN ;
    int w = fig->width -  (2.5 * p_w);
    int h = fig->height - (2 * p_h);

    priv->h = h;
    priv->w = w;
    priv->top = p_h;
    priv->left = 1.5 * p_w;
    priv->x0 = 1.5 * p_w;
    priv->y0 = h + p_h;

    fig->priv = priv;
    return priv;
}

static void figure_save_svg(figure_t* fig, const char* path) {
    if(!fig || !path) return;
    // 0. Open file & Compute priv of svg & Alloc buffer
    FILE *fptr = fopen(path, "w");
    if (!fptr) {
        LOG_ERR("Figure open %s fail!\n", path);
        return;
    }
    figure_priv_svg_t* priv = figure_priv_svg(fig);
    if(!priv) return;
    char* svg_buf = malloc(FIGURE_SVG_BUF_SIZE * sizeof(char));
    memset(svg_buf, 0, FIGURE_SVG_BUF_SIZE);
    // 1. Draw svg
    svg_header(svg_buf, fig->width, fig->height);
    svg_title(svg_buf, fig->title, priv->left, priv->w);
    svg_footer(svg_buf);
    // -1. Output to file & Close file & Free svg buffer
    fprintf(fptr, "%s", svg_buf);
    fclose(fptr);
    free(svg_buf);
}

// ==================================================================================== //
//                                  figure: API
// ==================================================================================== //

figure_t* figure_new(const char* title, figure_type_t type, size_t width, size_t height) {
    figure_t* fig = sys_malloc(sizeof(figure_t));
    fig->title = sys_strdup(title);
    fig->type = type;
    fig->width = width;
    fig->height = height;
    return fig;
}

void figure_save(figure_t* fig, const char* path) {
    if(!fig || !path) return;
    switch(fig->type) {
        case FIGURE_TYPE_BITMAP:
            break;
        case FIGURE_TYPE_VECTOR:
            figure_save_svg(fig, path);
            break;
        default:
            break;
    }
}

void figure_free(figure_t* fig) {
    if(!fig) return;
    if(fig->priv) {
        switch(fig->type) {
            case FIGURE_TYPE_BITMAP:
                break;
            case FIGURE_TYPE_VECTOR:
                figure_priv_svg_t* priv = fig->priv;
                free(priv);
                fig->priv = NULL;
                break;
            default:
                break;
        }
    }
    free(fig);
    fig = NULL;
}
