#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#include <string.h>


static void graph_init(graph_t *g, model_t *mdl) {
    g->name = NULL;
    g->tensors = NULL;
    g->nodes = NULL;
    g->input_inodes_vec = vector_create();
    g->output_inodes_vec = vector_create();

    g->ntensor = 0;
    g->nnode = 0;
    g->ninput_node = 0;
    g->noutput_node = 0;

    g->mdl = mdl;
    g->sez = mdl->sez;
    g->dev = mdl->dev;

    g->data_layout = 0;                             /* Default: NCHW */
    g->is_sub = 0;                                  /* Default: not  */
    g->status = GRAPH_STATUS_INIT;                  /* Default: INIT */

    g->sub_vec = vector_create();                   /* Sub graph vec */

    if(mdl) {
        mdl->graph = g;
        g->sez = mdl->sez;
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

    g->mdl = pg->mdl;
    g->sez = pg->sez;
    g->dev = pg->dev;

    g->data_layout = pg ? pg->data_layout : 0;
    g->is_sub = 1;                                  /* Default: yes  */
    g->status = GRAPH_STATUS_INIT;                  /* Default: INIT */

    g->idx = vector_size(pg->sub_vec);
    g->pgraph = pg;
    g->prof = NULL;

    if(pg->name) {
        char num[12];
        sprintf(num, "_%d", g->idx);
        g->name = strcat(sys_strdup(pg->name), num);
    } else {
        g->name = NULL;
    }

    vector_add(&pg->sub_vec, g);                    // Append sub graph to sub_vec[0]
}


graph_t * graph_new(model_t *mdl) {
    graph_t *g = (graph_t*)sys_malloc(sizeof(graph_t));
    if(!g) {
        return NULL;
    }
    graph_init(g, mdl);
    return g;
}

graph_t * graph_sub_new(graph_t* g) {
    graph_t *sg = (graph_t*)sys_malloc(sizeof(graph_t));
    if(!g || !sg) {
        return NULL;
    }
    graph_sub_init(sg, g);
    return sg;
}

graph_t * graph_as_sub(graph_t* g) {
    if(!g || g->is_sub) return g;
    graph_t *sg = (graph_t*)sys_malloc(sizeof(graph_t));
    if(!sg) {
        return NULL;
    }
    graph_sub_init(sg, g);
    for(int i = 0; i < sg->nnode; i++) {             // Copy All nodes from parent
        vector_add(&(sg->nodes_vec), i);
    }
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
    if(!g || !g->mdl) return;
    model_t *mdl = g->mdl;
    if(mdl->scd) {
        if(!g->is_sub && vector_size(g->sub_vec) == 0 ) {
            graph_as_sub(g);
        }
        mdl->scd->prerun(mdl->scd, g);
    }
}

void graph_step(graph_t* g, int n) {
    if(!g || !g->mdl) return;
    model_t *mdl = g->mdl;
    g->status = GRAPH_STATUS_RUN;
    mdl->scd->step(mdl->scd, g, n);
}

void graph_run(graph_t *g) {
    if(!g || !g->mdl) return;
    model_t *mdl = g->mdl;
    g->status = GRAPH_STATUS_RUN;
    mdl->scd->run(mdl->scd, g);
}

void graph_wait(graph_t *g) {
    if(!g || !g->mdl) return;
    model_t *mdl = g->mdl;
    g->status = GRAPH_STATUS_SUSPEND;
    mdl->scd->wait(mdl->scd, g);
}

void graph_posrun(graph_t *g) {
    if(!g || !g->mdl) return;
    model_t *mdl = g->mdl;
    mdl->scd->posrun(mdl->scd, g);
}

void graph_dump(graph_t* g) {
    if(!g) return;
    LOG_INFO("[Graph: %s]\n", g->name);
    LOG_INFO("| --------------------------------------------------------- |\n");
    LOG_INFO("|     Layers(%3d)      |      Input      |      Output      |\n", g->nnode);
    LOG_INFO("| -------------------- | --------------- | ---------------- |\n");
    for(int i=0; i < g->nnode; i++) {
        if(g->nodes[i]) {
            char* in = g->nodes[i]->in ? tensor_dump_shape(g->nodes[i]->in[0]) : sys_strdup("[]");
            char* out = g->nodes[i]->out ? tensor_dump_shape(g->nodes[i]->out[0]) : sys_strdup("[]");
            LOG_INFO("| %20s | %15s | %16s |\n", g->nodes[i]->op ? op_name(g->nodes[i]->op->type) : NULL, in, out);
            free(in);
            free(out);
        }
    }
    LOG_INFO("| --------------------------------------------------------- |\n");
}

// Atterntion: Node's Operate Type May Be Changed After `graph_prerun`
void graph_dump2(graph_t* g) {
    if(!g) return;
    LOG_INFO("[Graph: %s]\n", g->name);
    for(int i=0; i < g->nnode; i++) {
        node_dump(g->nodes[i]);
    }
}

void graph_exec_report(graph_t *g) {
    graph_exec_report_level(g, 0);
}

void graph_exec_report_level(graph_t *g, int l) {
    if(g->is_sub && g->prof) {
        profiler_report(g->prof, l);
    } else if(!g->is_sub) {
        for(int i = 0; i < vector_size(g->sub_vec); i++) {
            graph_exec_report_level(g->sub_vec[i], l);
        }
    }
}

void graph_free(graph_t *g) {
    if(!g) return;
    // free tensors
    if(g->tensors) {
        for(int i = 0; i < g->ntensor; i++){
            if(g->tensors[i]) {
                tensor_free(g->tensors[i]);
                g->tensors[i] = NULL;
            }
        }
        if(g->tensors) sys_free(g->tensors);
        g->tensors = NULL;
    }
    // free nodes
    if(g->nodes) {
        for(int i = 0; i < g->nnode; i++){
            if(g->nodes[i]) {
                node_free(g->nodes[i]);
                g->nodes[i] = NULL;
            }
        }
        if(g->nodes) sys_free(g->nodes);
        g->nodes = NULL;
    }
    if(g->name) sys_free(g->name);
    if(g->is_sub) {
        if(g->prof) profiler_free(g->prof);
        if(g->nodes_vec) { vector_free(g->nodes_vec); g->nodes_vec = NULL; }
        if(g->input_itensors_vec) { vector_free(g->input_itensors_vec); g->input_inodes_vec = NULL; }
        if(g->output_itensors_vec) { vector_free(g->output_itensors_vec); g->output_itensors_vec = NULL; }
    } else {
        if(g->sub_vec) {
            vector_free(g->sub_vec); 
            g->sub_vec = NULL; 
        }
        if(g->input_inodes_vec) { vector_free(g->input_inodes_vec); g->input_inodes_vec = NULL; }
        if(g->output_inodes_vec) { vector_free(g->output_inodes_vec); g->output_inodes_vec = NULL; }
    }
    if(g) sys_free(g);
    g = NULL;
}