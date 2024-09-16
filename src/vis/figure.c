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
    float *     vals;
    float *     pos;
    char **     labels;
    size_t      n;
    int         t_exp;
    int         tick;
} svg_ticks_t;

typedef struct {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
    float x0;
    float y0;
    float top;
    float left;
    float w;
    float h;
} figure_priv_svg_t;


static void svg_title(char* buffer, char* title, float left, float width) {
    if(!buffer) return;
    float tx = left + width / 2.0f;
    float ty = 15.0f;
    svg_text_bold(buffer, tx, ty, SVG_TXT_MIDDLE, title ? title : "title", NULL);
}

static void svg_axis2d(char* buffer, figure_t* fig) {
    if(!buffer || !fig || !fig->priv) return;
    figure_priv_svg_t* priv = fig->priv;
    int p_h = priv->top;
    int p_w = priv->left;
    int w = priv->w;
    int h = priv->h;
    svg_rectangle(buffer, p_w, p_h, w, h, "none", "black", 0.5, NULL);
    if(fig->axiss[0]->label != NULL) {
        float tx = p_w + priv->w / 2;
        float ty = FIGURE_SVG_Y_MARGIN + priv->y0 + FIGURE_SVG_Y_MARGIN / 2;
        svg_text_regular(buffer, tx, ty, SVG_TXT_MIDDLE, fig->axiss[0]->label, NULL);
    }
    if(fig->axiss[1]->label != NULL) {
        float tx = 5;
        float ty = p_h + priv->h / 2;
        char * transform = malloc(64 * sizeof(char));
        memset(transform, 0, 64 * sizeof(char));
        sprintf(transform, "rotate(270, %.2f, %.2f) translate(0, 10)", tx, ty);
        svg_text_transform(buffer, tx, ty, SVG_TXT_MIDDLE, SVG_TXT_NORMAL, transform, fig->axiss[1]->label, NULL);
        free(transform);
    }
}

static void svg_legend(char* buffer, figure_t*fig) {
    if(!buffer || !fig || !fig->plot_vec || !fig->priv) return;
    for(int i = 0; i < vector_size(fig->plot_vec); i++) {
        
    }
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
    int w = fig->width -  (fig->axiss[1]->label ? 3.0f * p_w : 2.5f * p_w);
    int h = fig->height - (fig->axiss[0]->label ? 3.0f * p_h : 2.0f * p_h);

    priv->h = h;
    priv->w = w;
    priv->top = p_h;
    priv->left = fig->axiss[1]->label ? 2.0f * p_w : 1.5f * p_w;
    priv->x0 = fig->axiss[1]->label ? 2.0f * p_w : 1.5f * p_w;
    priv->y0 = h + p_h;

    if(fig->axiss[0]->type == FIGURE_AXIS_TYPE_LINEAR) {
        priv->x_max = figure_get_max(fig, 0);
        priv->x_min = figure_get_min(fig, 0);
    } else {
        priv->x_max = log10f(figure_get_max(fig, 0)) + 0.05;
        priv->x_min = log10f(figure_get_min(fig, 0)) + 0.05;
    }

    if(fig->axiss[1]->type == FIGURE_AXIS_TYPE_LINEAR) {
        priv->y_max = figure_get_max(fig, 1);
        priv->y_min = figure_get_min(fig, 1);
    } else {
        priv->y_max = log10f(figure_get_max(fig, 1)) + 0.05;
        priv->y_min = log10f(figure_get_min(fig, 1)) + 0.05;
    }

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
    svg_clip_region(svg_buf, priv->left, priv->top, priv->w, priv->h, "plot-area");
    svg_title(svg_buf, fig->title, priv->left, priv->w);
    svg_axis2d(svg_buf, fig);
    svg_footer(svg_buf);
    // -1. Output to file & Close file & Free svg buffer
    fprintf(fptr, "%s", svg_buf);
    fclose(fptr);
    free(svg_buf);
}

// ==================================================================================== //
//                                  figure: API
// ==================================================================================== //

figure_axis_t* figure_axis_new(char* label, figure_axis_type_t type, bool auto_scale) {
    figure_axis_t* fig_axis = malloc(sizeof(figure_axis_t));
    memset(fig_axis, 0, sizeof(figure_axis_t));
    fig_axis->label = label != NULL ? sys_strdup(label) : NULL;
    fig_axis->type = type;
    fig_axis->is_auto_scale = auto_scale;
    fig_axis->range_min = 0.0;
    fig_axis->range_max = 0.0;
    return fig_axis;
}

void figure_axis_set_label(figure_axis_t* fig_axis, char* label) {
    if(!fig_axis) return;
    if(fig_axis->label) free(fig_axis->label);
    fig_axis->label = label ? sys_strdup(label) : NULL;
}

void figure_axis_free(figure_axis_t* fig_axis) {
    if(fig_axis) {
        if(fig_axis->label) {
            free(fig_axis->label); 
            fig_axis->label = NULL;
        }
        free(fig_axis);
        fig_axis = NULL;
    }
}


figure_plot_t* figure_plot_new(char* label, figure_plot_type_t type, tensor_t* data) {
    figure_plot_t* fig_plot = malloc(sizeof(figure_plot_t));
    memset(fig_plot, 0, sizeof(figure_plot_t));
    fig_plot->label = label != NULL ? sys_strdup(label) : NULL;
    fig_plot->type = type;
    fig_plot->data = data;
    return fig_plot;
}

/**
 * fig_plot->data:
 * - type: float32
 * - ndim: 2
 * - dims: [plot_num, axis_num]
 * 
 *      0   1   2   3
 *  0  x0  y0  z0  ..
 *  1  x1  y1  z1  ..
 *  2  x2  y2  z2  ..
 *  3  x3  y3  z3  ..
 *  4  ..  ..  ..  ..
 * 
 */

int figure_plot_number(figure_plot_t* fig_plot) {
    if(!fig_plot || !fig_plot->data || fig_plot->data->ndim != 2) return 0;
    return fig_plot->data->dims[0];
}

float figure_plot_get_max(figure_plot_t* fig_plot, size_t n) {
    if(!fig_plot || !fig_plot->data || fig_plot->data->ndim != 2) return 0.0f;
    int nplot = fig_plot->data->dims[0];
    if(nplot <= 0) return 0.0f;
    int naxis = fig_plot->data->dims[1];
    float* data = fig_plot->data->datas;
    float max_n = data[n];
    for(int i = 1; i < nplot; i++) {
        if(data[n + i * naxis] > max_n) {
            max_n = data[n + i * naxis];
        }
    }
    return max_n;
}

float figure_plot_get_min(figure_plot_t* fig_plot, size_t n) {
    if(!fig_plot || !fig_plot->data || fig_plot->data->ndim != 2) return 0.0f;
    int nplot = fig_plot->data->dims[0];
    if(nplot <= 0) return 0.0f;
    int naxis = fig_plot->data->dims[1];
    float* data = fig_plot->data->datas;
    float min_n = data[n];
    for(int i = 1; i < nplot; i++) {
        if(data[n + i * naxis] < min_n) {
            min_n = data[n + i * naxis];
        }
    }
    return min_n;
}

void figure_plot_free(figure_plot_t* fig_plot) {
    if(fig_plot) {
        if(fig_plot->label) {
            free(fig_plot->label); 
            fig_plot->label = NULL;
        }
        if(fig_plot->data) {
            tensor_free(fig_plot->data);
            fig_plot->data = NULL;
        }
        free(fig_plot);
        fig_plot = NULL;
    }
}

figure_t* figure_new(const char* title, figure_type_t type, size_t width, size_t height, size_t naxis) {
    figure_t* fig = sys_malloc(sizeof(figure_t));
    fig->title = sys_strdup(title);
    fig->type = type;
    fig->width = width;
    fig->height = height;

    fig->naxis = naxis;
    fig->axiss = malloc(sizeof(figure_axis_t*) * naxis);
    for(int i = 0; i < naxis; i++) {
        fig->axiss[i] = figure_axis_new(NULL, FIGURE_AXIS_TYPE_LINEAR, true);
    }

    fig->plot_vec = vector_create();

    return fig;
}

void figure_add_plot(figure_t* fig, figure_plot_t* plot) {
    if(!fig || !plot) return;
    vector_add(&fig->plot_vec, plot);
}

figure_plot_t* figure_get_plot(figure_t* fig, size_t i) {
    if(!fig || !fig->plot_vec || i >= vector_size(fig->plot_vec)) return NULL;
    return fig->plot_vec[i];
}

float figure_get_max(figure_t* fig, size_t n) {
    if(!fig || n >= fig->naxis || !fig->plot_vec) return 0.0f;
    if(fig->axiss[n]->is_auto_scale) {
        if(vector_size(fig->plot_vec) <= 0) return 0.0f;
        float max_n = figure_plot_get_max(fig->plot_vec[0], n);
        float min_n = figure_plot_get_min(fig->plot_vec[0], n);
        for(int i = 1; i < vector_size(fig->plot_vec); i++) {
            figure_plot_t* p = figure_get_plot(fig, i);
            if(p && figure_plot_number(p) > 0) {
                float max_i = figure_plot_get_max(p, n);
                float min_i = figure_plot_get_min(p, n);
                if(max_i > max_n) max_n = max_i;
                if(min_i < min_n) min_n = min_i;
            }
        }
        if(fig->axiss[n] == FIGURE_AXIS_TYPE_LINEAR){
            max_n += (max_n - min_n) * 0.05f;
        }
        return max_n;
    }
    return fig->axiss[n]->range_max;
}

float figure_get_min(figure_t* fig, size_t n) {
    if(!fig || n >= fig->naxis || !fig->plot_vec) return 0.0f;
    if(fig->axiss[n]->is_auto_scale) {
        if(vector_size(fig->plot_vec) <= 0) return 0.0f;
        float max_n = figure_plot_get_max(fig->plot_vec[0], n);
        float min_n = figure_plot_get_min(fig->plot_vec[0], n);
        for(int i = 1; i < vector_size(fig->plot_vec); i++) {
            figure_plot_t* p = figure_get_plot(fig, i);
            if(p && figure_plot_number(p) > 0) {
                float max_i = figure_plot_get_max(p, n);
                float min_i = figure_plot_get_min(p, n);
                if(max_i > max_n) max_n = max_i;
                if(min_i < min_n) min_n = min_i;
            }
        }
        if(fig->axiss[n] == FIGURE_AXIS_TYPE_LINEAR){
            min_n -= (max_n - min_n) * 0.05f;
        }
        return min_n;
    }
    return fig->axiss[n]->range_min;
}

void figure_save(figure_t* fig, const char* path) {
    if(!fig || !path) return;
    switch(fig->type) {
        case FIGURE_TYPE_BITMAP:
            break;
        case FIGURE_TYPE_VECTOR:
            figure_save_svg(fig, path);
            LOG_INFO("Figure save: %s\n", path);
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
                break;
            default:
                break;
        }
        fig->priv = NULL;
    }
    if(fig->axiss) {
        for(int i = 0; i < fig->naxis; i++) {
            if(fig->axiss[i]) figure_axis_free(fig->axiss[i]);
        }
        fig->axiss = NULL;
        fig->naxis = 0;
    }
    if(fig->plot_vec) {
        vector_free(fig->plot_vec);
        fig->plot_vec = NULL;
    }
    free(fig);
    fig = NULL;
}
