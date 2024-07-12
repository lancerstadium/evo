#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"


static void node_init(node_t* nd, op_type_t op_ty, int nd_idx) {
    nd->index = nd_idx;
    nd->nin = 0;
    nd->nout= 0;
    nd->type = NODE_TYPE_INTER;
    nd->in = NULL;
    nd->out = NULL;
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
    if(name) {
        nd->name = sys_strdup(name);
    }
    return nd;
}

void node_dump(node_t *nd) {
    int i;
    if(nd) {
        LOG_INFO("%s-%d: %s\r\n", nd->op->name ? nd->op->name : "Uninit" , nd->opset, nd->name);
        if(nd->nin > 0) {
            LOG_INFO("  - Inputs: \n");
            for(i = 0; i < nd->nin; i++) {
                LOG_INFO("        ");
                tensor_dump(nd->in[i]);
            }
        }
        if(nd->nout > 0) {
            LOG_INFO("  - Outputs: \n");
            for(i = 0; i < nd->nout; i++) {
                LOG_INFO("        ");
                tensor_dump(nd->out[i]);
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
    if(nd->nin > 0) {
        sys_free(nd->in);
        nd->in= NULL;
    }
    if(nd->nout > 0) {
        sys_free(nd->out);
        nd->out = NULL;
    }
    sys_free(nd);
}