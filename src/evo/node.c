
#include "evo.h"
#include "sys.h"


static void node_init(node_t* nd, op_type_t op_ty, int nd_idx) {
    nd->index = nd_idx;
    nd->ninput = 0;
    nd->noutput= 0;
    nd->type = EVO_NODE_TYPE_INTERMEDIATE;
    nd->input_tensors = NULL;
    nd->output_tensors = NULL;
    nd->name = NULL;
    nd->op.type = op_ty;
    nd->op.is_same_shape = 1;
    nd->op.param_size = 0;
    nd->op.param_mem = NULL;
    nd->op.infer_shape = NULL;
    nd->op.init = NULL;
    nd->op.release = NULL;
}

node_t * node_new(graph_t* g, const char* name, op_type_t op_ty) {
    node_t * nd = (node_t*)sys_malloc(sizeof(node_t));
    if(nd == NULL) {
        return NULL;
    }
    node_init(nd, op_ty, g->nnode);

    node_t ** new_node_list = (node_t**)sys_realloc(g->nodes, sizeof(node_t*) * (g->nnode + 1));
    if(new_node_list == NULL) {
        sys_free(nd);
        return NULL;
    }
    nd->graph = g;
    if(name) {
        nd->name = sys_strdup(name);
    }
    new_node_list[g->nnode] = nd;
    g->nodes = new_node_list;
    g->nnode++;
    return nd;
}

void node_free(node_t* nd, graph_t* g) {
    if(!nd || !g) {
        return;
    }
    if(nd->name) {
        sys_free(nd->name);
        nd->name = NULL;
    }
    if(nd->ninput > 0) {
        sys_free(nd->input_tensors);
        nd->input_tensors= NULL;
    }
    if(nd->noutput > 0) {
        sys_free(nd->output_tensors);
        nd->output_tensors = NULL;
    }
    if(nd->op.release) {
        nd->op.release(&nd->op);
    }
    sys_free(nd);
}