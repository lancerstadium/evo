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
    int w = fig->width -  (2.5f * p_w);
    int h = fig->height - (2.0f * p_h);

    priv->h = h;
    priv->w = w;
    priv->top = p_h;
    priv->left = 1.5f * p_w;
    priv->x0 = 1.5f * p_w;
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

figure_t* figure_add_plot(figure_t* fig, figure_plot_t* plot) {
    if(!fig || !plot) return;
    vector_add(&fig->plot_vec, plot);
}

figure_plot_t* figure_get_plot(figure_t* fig, size_t i) {
    if(!fig || !fig->plot_vec || i >= vector_size(fig->plot_vec)) return NULL;
    return fig->plot_vec[i];
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
