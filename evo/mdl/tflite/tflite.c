#include "tflite.h"
#include "tflite_builder.h"
#include "tflite_reader.h"
#include "../../util/sys.h"
#include "../../util/log.h"
#include <stdio.h>


#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(tflite, x)  // Specified in the schema.

EVO_UNUSED static flatcc_builder_t builder;

context_t *load_tflite(struct serializer *s, const void *buf, size_t len) {
    // flatcc_builder_init(&builder);
    context_t *ctx = NULL;
    if (!buf || len <= 0)
        return NULL;
    ctx = context_new(NULL);
    ctx->sez = s;
    ctx->cmodel = ns(Model_as_root(buf));
    if (!ctx->cmodel) {
        if (ctx)
            sys_free(ctx);
        return NULL;
    }
    ctx->model_size = len;
    ctx->name = sys_strdup(ns(Model_description(ctx->cmodel)));
    ctx->tensor_map = hashmap_create();
    if (!ctx->tensor_map) {
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
    if (ctx && ctx->cmodel) {
        // flatcc_builder_clear(&builder);
        ctx->model_size = 0;
    }
}

EVO_UNUSED tensor_t *load_tensor_tflite(const char *path);

graph_t *load_graph_tflite(context_t *ctx) {
    if (!ctx || !ctx->cmodel) {
        return NULL;
    }
    graph_t *g;
    ns(SubGraph_vec_t) subgraphs = ns(Model_subgraphs(ctx->cmodel));
    if (!subgraphs)
        return NULL;
    g = graph_new(ctx);
    if (!g)
        return NULL;
    // Print cur Operator codes
    ns(OperatorCode_vec_t) opcodes = ns(Model_operator_codes(ctx->cmodel));
    for(size_t i = 0; i < ns(OperatorCode_vec_len(opcodes)); i++) {
        ns(OperatorCode_table_t) opcode = ns(OperatorCode_vec_at(opcodes, i));
        LOG_INFO("opcode: %d\n", ns(OperatorCode_builtin_code(opcode)));
    }
    // New Subgraph: subgraph <== tflite subgraph
    for(size_t i = 0; i < ns(SubGraph_vec_len(subgraphs)); i++) {
        ns(SubGraph_table_t) subgraph = ns(SubGraph_vec_at(subgraphs, i));
        graph_t *sg = graph_sub_new(g);
        LOG_INFO("subgraph: %s\n", ns(SubGraph_name(subgraph)));
        // Add Tensors
        ns(Tensor_vec_t) tensors = ns(SubGraph_tensors(subgraph));
        sg->ntensor = ns(Tensor_vec_len(tensors));
        for(size_t j = 0; j < sg->ntensor; j++) {
            ns(Tensor_table_t) tensor = ns(Tensor_vec_at(tensors, j));
        }
        vector_add(&(g->sub_vec), sg);
    }
    return g;
}

EVO_UNUSED int save_tflite() {
    return 0;
}


#undef ns