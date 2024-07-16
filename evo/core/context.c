#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#include <string.h>


context_t * context_new(const char *name) {
    context_t * ctx = (context_t*)sys_malloc(sizeof(context_t));
    if(name) {
        ctx->name = sys_strdup(name);
    } else {
        ctx->name = NULL;
    }
    ctx->sez = NULL;
    ctx->scd = scheduler_get_default();         /* Default: sync scheduler  */
    ctx->dev = device_registry_find("cpu");     /* Default: device cpu      */
    // load model
    ctx->model = NULL;
    ctx->model_size = 0;
    // init graph
    ctx->graph = graph_new(ctx);
    // init tensor map
    ctx->tensor_map = hashmap_create();
    return ctx;
}


tensor_t* context_get_tensor(context_t *ctx, const char *name) {
    if(ctx && ctx->tensor_map) {
        tensor_t * t = NULL;
        int res = hashmap_get(ctx->tensor_map, hashmap_str_lit(name), (uintptr_t*)&t);
        if(res != 0 && t && strcmp(name, t->name) == 0) {
            return t;
        }
    }
    return NULL;
}


void context_free(context_t *ctx) {
    if(ctx) {
        if(ctx->name) free(ctx->name);
        if(ctx->model) ctx->sez->unload(ctx);
        if(ctx->tensor_map) hashmap_free(ctx->tensor_map);
    }
    ctx = NULL;
}