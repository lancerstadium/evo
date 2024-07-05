

#include "evo.h"
#include "sys.h"


context_t * context_new(const char *name) {
    context_t * ctx = (context_t*)sys_malloc(sizeof(context_t));
    if(name) {
        ctx->name = sys_strdup(name);
    } else {
        ctx->name = NULL;
    }
    ctx->scd = NULL;
    ctx->dev = NULL;
    return ctx;
}


void context_free(context_t *ctx) {
    free(ctx->name);
    ctx = NULL;
}