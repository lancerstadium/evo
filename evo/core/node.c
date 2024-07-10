#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"


static void node_init(node_t* nd, op_type_t op_ty, int nd_idx) {
    nd->index = nd_idx;
    nd->ninput = 0;
    nd->noutput= 0;
    nd->type = NODE_TYPE_INTER;
    nd->input_tensors = NULL;
    nd->output_tensors = NULL;
    nd->name = NULL;
    // operater
    nd->op = (op_t*)sys_malloc(sizeof(op_t));
    nd->op->type = op_ty;
    nd->op->is_same_shape = 1;
    nd->op->param_size = 0;
    nd->op->param_mem = NULL;
}

node_t * node_new(graph_t* g, const char* name, op_type_t op_ty) {
    node_t * nd = (node_t*)sys_malloc(sizeof(node_t));
    if(nd == NULL) {
        return NULL;
    }
    node_init(nd, op_ty, g->nnode);
    nd->opset = 0;
    nd->graph = g;
    nd->ctx = g->ctx;
    nd->node_proto = NULL;
    nd->reshape = NULL;
    nd->operator = NULL;
    if(name) {
        nd->name = sys_strdup(name);
    }
    return nd;
}

void node_dump(node_t *nd) {
    int i;
    if(nd) {
        LOG_INFO("%s-%d: %s\r\n", nd->op->name ? nd->op->name : "Uninit" , nd->opset, nd->name);
        if(nd->ninput > 0) {
            LOG_INFO("  - Inputs: \n");
            for(i = 0; i < nd->ninput; i++) {
                LOG_INFO("        ");
                tensor_dump(nd->input_tensors[i]);
            }
        }
        if(nd->noutput > 0) {
            LOG_INFO("  - Outputs: \n");
            for(i = 0; i < nd->noutput; i++) {
                LOG_INFO("        ");
                tensor_dump(nd->output_tensors[i]);
            }
        }
    }
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
    sys_free(nd);
}