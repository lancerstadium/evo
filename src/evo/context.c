

#include "evo.h"
#include "sys.h"


context_t * context_new(const char *name) {
    context_t * ctx = (context_t*)sys_malloc(sizeof(context_t));
    if(name) {
        ctx->name = sys_strdup(name);
    } else {
        ctx->name = NULL;
    }
    ctx->sez = NULL;
    ctx->scd = NULL;
    ctx->dev = NULL;
    // load model
    ctx->model = NULL;
    ctx->model_size = 0;
    // init graph
    ctx->graph = graph_new(ctx);
    return ctx;
}


void context_free(context_t *ctx) {
    free(ctx->name);
    ctx = NULL;
}