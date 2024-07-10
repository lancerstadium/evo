

#include "evo.h"
#include "log.h"
#include "sys.h"


static void graph_init(graph_t *g, context_t *ctx) {
    g->tensors = NULL;
    g->nodes = NULL;
    g->input_inodes_vec = vector_create();
    g->output_inodes_vec = vector_create();

    g->ntensor = 0;
    g->nnode = 0;
    g->ninput_node = 0;
    g->noutput_node = 0;

    g->ctx = ctx;
    g->sez = ctx->sez;
    g->dev = ctx->dev;

    g->data_layout = 0;                             /* Default: NCHW */
    g->is_sub = 0;                                  /* Default: not  */
    g->status = GRAPH_STATUS_INIT;                  /* Default: INIT */

    g->sub_vec = vector_create();                   /* Sub graph vec */

    if(ctx) {
        ctx->graph = g;
        g->sez = ctx->sez;
    }
}

static void graph_sub_init(graph_t *g, graph_t *pg) {
    g->tensors = pg->tensors;
    g->nodes = pg->nodes;
    g->ntensor = pg->ntensor;
    g->nnode = pg->nnode;
    g->nodes_vec = vector_create();
    g->input_itensors_vec = vector_create();
    g->output_itensors_vec = vector_create();

    g->ctx = pg->ctx;
    g->sez = pg->sez;
    g->dev = pg->dev;

    g->data_layout = pg ? pg->data_layout : 0;
    g->is_sub = 1;                                  /* Default: yes  */
    g->status = GRAPH_STATUS_INIT;                  /* Default: INIT */

    g->idx = 0;                                     /* Need to modify */
    g->pgraph = pg;
    g->info = NULL;

    for(int i = 0; i < g->nnode; i++) {             // Copy nodes from parent
        vector_add(&(g->nodes_vec), i);
    }
    vector_add(&pg->sub_vec, g);                    // Append sub graph to sub_vec[0]
}


graph_t * graph_new(context_t *ctx) {
    graph_t *g = (graph_t*)sys_malloc(sizeof(graph_t));
    if(!g) {
        return NULL;
    }
    graph_init(g, ctx);
    return g;
}

graph_t * graph_sub(graph_t* g) {
    graph_t *sg = (graph_t*)sys_malloc(sizeof(graph_t));
    if(!sg) {
        return NULL;
    }
    graph_sub_init(sg, g);
    return sg;
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


void graph_push_node(graph_t* g, node_t* nd) {
    if(nd && g) {
        nd->index = g->nnode;
        // nd list
        node_t ** new_node_list = (node_t **)sys_realloc(g->nodes, (g->nnode + 1) * sizeof(node_t *));
        if(!new_node_list) return;
        // update
        new_node_list[g->nnode] = nd;
        g->nodes = new_node_list;
        g->nnode++;
    }
}

node_t* graph_get_node(graph_t *g, int i) {
    if(g && (i >= 0) && (i < g->nnode)) {
        return g->nodes[i];
    }
    return NULL;
}

void graph_prerun(graph_t *g) {
    if(!g || !g->ctx) return;
    context_t *ctx = g->ctx;
    if(ctx->scd) {
        if(!g->is_sub && vector_size(g->sub_vec) == 0 ) {
            graph_sub(g);
        }
        ctx->scd->prerun(ctx->scd, g);
    }
}

void graph_step(graph_t* g, int n) {
    if(!g || !g->ctx) return;
    context_t *ctx = g->ctx;
    g->status = GRAPH_STATUS_RUN;
    ctx->scd->step(ctx->scd, g, n);
}

void graph_run(graph_t *g) {
    if(!g || !g->ctx) return;
    context_t *ctx = g->ctx;
    g->status = GRAPH_STATUS_RUN;
    ctx->scd->run(ctx->scd, g);
}

void graph_wait(graph_t *g) {
    if(!g || !g->ctx) return;
    context_t *ctx = g->ctx;
    g->status = GRAPH_STATUS_SUSPEND;
    ctx->scd->wait(ctx->scd, g);
}

void graph_posrun(graph_t *g) {
    if(!g || !g->ctx) return;
    context_t *ctx = g->ctx;
    ctx->scd->posrun(ctx->scd, g);
}

// Atterntion: Node's Operate Type May Be Changed After `graph_prerun`
void graph_dump(graph_t* g) {
    if(!g) return;
    LOG_INFO("[Graph]\n");
    for(int i=0; i < g->nnode; i++) {
        node_dump(g->nodes[i]);
    }
}

void graph_free(graph_t *g) {
    if(!g) return;
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

    if(g->is_sub) {
        if(g->info) sys_free(g->info);
        if(g->nodes_vec) vector_free(g->nodes_vec);
        if(g->input_itensors_vec) vector_free(g->input_itensors_vec);
        if(g->output_itensors_vec) vector_free(g->output_itensors_vec);
    } else {
        if(g->sub_vec) vector_free(g->sub_vec);
        if(g->input_inodes_vec) vector_free(g->input_inodes_vec);
        if(g->output_inodes_vec) vector_free(g->output_inodes_vec);
    }

    g = NULL;
}