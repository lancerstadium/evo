

#include "evo.h"
#include "sys.h"


static void graph_init(graph_t *g, context_t *ctx) {
    g->tensors = NULL;
    g->nodes = NULL;
    g->input_nodes = NULL;
    g->output_nodes = NULL;

    g->ntensor = 0;
    g->nnode = 0;
    g->ninput_node = 0;
    g->noutput_node = 0;

    g->ctx = ctx;
    g->sez = NULL;
    g->dev = NULL;

    g->data_layout = 0;             /* Default: NCHW */
    g->is_sub = 0;                  /* Default: not  */
    g->status = GRAPH_STATUS_INIT;  /* Default: INIT */

    if(ctx) {
        ctx->graph = g;
        g->sez = ctx->sez;
    }
}

graph_t * graph_new(context_t *ctx) {
    graph_t *g = (graph_t*)sys_malloc(sizeof(graph_t));
    if(!g) {
        return NULL;
    }
    graph_init(g, ctx);
    return g;
}

void graph_push_tenser(graph_t* g, tensor_t* ts) {
    if(ts && g) {
        ts->layout = g->data_layout;
        ts->index = g->ntensor;
        // ts list
        tensor_t ** new_tensor_list = (tensor_t **)sys_realloc(g->tensors, (g->ntensor + 1) * sizeof(tensor_t *));
        if(!new_tensor_list) return;
        // update
        new_tensor_list[g->ntensor] = ts;
        g->tensors = new_tensor_list;
        g->ntensor++;
    }
}

void graph_prerun(graph_t *g) {
    context_t *ctx = g->ctx;
    scheduler_t *scd = ctx->scd;
    scd->prerun(scd, g);
}

void graph_run(graph_t *g) {
    context_t *ctx = g->ctx;
    scheduler_t *scd = ctx->scd;
    g->status = GRAPH_STATUS_RUN;
    scd->run(scd, g);
}

void graph_wait(graph_t *g) {
    context_t *ctx = g->ctx;
    scheduler_t *scd = ctx->scd;
    scd->wait(scd, g);
}

void graph_posrun(graph_t *g) {
    context_t *ctx = g->ctx;
    scheduler_t *scd = ctx->scd;
    scd->posrun(scd, g);
}

void graph_free(graph_t *g) {

    // free tensors
    for(int i = 0; i < g->ntensor; i++) {
        tensor_free(g->tensors[i]);
    }
    // free nodes
    for(int i = 0; i < g->nnode; i++) {
        node_free(g->nodes[i], g);
    }
    sys_free(g->tensors);
    sys_free(g->nodes);

    g = NULL;
}