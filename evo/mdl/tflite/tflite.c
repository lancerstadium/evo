#include "tflite.h"
#include "tflite_builder.h"
#include "tflite_reader.h"
#include "../../util/sys.h"
#include "../../util/log.h"
#include <stdio.h>


#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(etm, x)  // Specified in the schema.

static flatcc_builder_t builder;

static int model_create(flatcc_builder_t *B) {
    return 0;
}

context_t *load_tflite(struct serializer *s, const void *buf, size_t len) {
    flatcc_builder_init(&builder);
    context_t *ctx = NULL;
    if (!buf || len <= 0)
        return NULL;
    ctx = context_new(NULL);
    ctx->sez = s;
    model_create(&builder);
    ctx->model = flatcc_builder_finalize_aligned_buffer(&builder, &len);
    ctx->model_size = len;
    ctx->name = sys_strdup("tflite");
    if (!ctx->model) {
        if (ctx)
            sys_free(ctx);
        return NULL;
    }
    ctx->tensor_map = hashmap_create();
    if (!ctx->tensor_map) {
        if (ctx->model)
            sys_free(ctx->model);
        if (ctx)
            sys_free(ctx);
        return NULL;
    }
    // graph
    load_graph_tflite(ctx);
    return ctx;
}


context_t *load_model_tflite(struct serializer *sez, const char *path) {
    context_t *ctx = NULL;
    FILE *fp;
    uint32_t len;
    unsigned int i;
    void *buf;
    fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        len = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        if (len > 0) {
            buf = sys_malloc(len);
            if (buf) {
                for (i = 0; i < len; i += fread(buf + i, 1, len - i, fp));
                ctx = load_tflite(sez, buf, len);
                sys_free(buf);
            }
        }
        fclose(fp);
    } else {
        LOG_ERR("No such file: %s\n", path);
    }
    return ctx;
}

void unload_tflite(context_t *ctx) {
    if (ctx && ctx->model) {
        flatcc_builder_aligned_free(ctx->model);
        flatcc_builder_clear(&builder);
        sys_free(ctx->model);
        ctx->model_size = 0;
    }
}

EVO_UNUSED tensor_t *load_tensor_tflite(const char *path);

graph_t *load_graph_tflite(context_t *ctx) {
    if (!ctx || !ctx->model) {
        return NULL;
    }
    graph_t *g;

    g = graph_new(ctx);
    if (!g)
        return NULL;

    return g;
}






#undef ns