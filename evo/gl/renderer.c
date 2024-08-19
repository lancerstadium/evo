#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../evo.h"
#include "../util/log.h"
#include "../util/math.h"
#include "../util/sys.h"


// ==================================================================================== //
//                                  renderer: gif
// ==================================================================================== //

typedef struct {
    char* name;
} renderer_gif_t;

void renderer_render_gif(renderer_t* rd, render_fn_t rd_fn) {
    if (!rd || !rd->priv || !rd_fn) return;
    renderer_gif_t* priv = rd->priv;
    canvas_t* cav = rd_fn(0);
    int64_t ndelay = 60;       // 0 < ndelay <= 60
    int64_t delays[ndelay];
    int64_t delay = 5;         // 5 ms
    delays[0] = delay;
    for (size_t i = 1; i < ndelay; i++) {
        delays[i] = delay;
        canvas_t* cav_tmp = rd_fn((1.f * i) / 60.f);
        image_push(cav->background, cav_tmp->background);
        // canvas_free(cav_tmp);
    }
    image_set_deloys(cav->background, delays, ndelay);
    canvas_export(cav, priv->name);
    canvas_free(cav);
}

// ==================================================================================== //
//                                  renderer: API
// ==================================================================================== //

renderer_t* renderer_new(renderer_type_t type) {
    renderer_t* rd = malloc(sizeof(renderer_t));
    rd->type = type;
    switch (type) {
        case RENDERER_TYPE_GIF: {
            renderer_gif_t* priv = malloc(sizeof(renderer_gif_t));
            if (priv) {
                priv->name = strdup("renderer.gif");
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
            switch(rd->type) {
                case RENDERER_TYPE_GIF: {
                    renderer_gif_t* priv = rd->priv;
                    if(priv->name) free(priv->name);
                }
                default: break;
            }
            free(rd->priv);
        }
        free(rd);
        rd = NULL;
    }
}
