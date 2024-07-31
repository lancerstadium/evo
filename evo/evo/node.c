#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#include <string.h>

static void node_init(node_t* nd, op_type_t op_ty, int nd_idx) {
    nd->index = nd_idx;
    nd->nin = 0;
    nd->nout= 0;
    nd->type = NODE_TYPE_HIDDEN;
    nd->in = NULL;
    nd->out = NULL;
    nd->name = NULL;
    nd->priv = NULL;
    // operater
    nd->op = (op_t*)sys_malloc(sizeof(op_t));
    nd->op->type = op_ty;
    nd->op->is_same_shape = 1;
    // attribute
    nd->attr_vec = vector_create();
}

node_t * node_temp(const char* name, op_type_t op_ty) {
    node_t * nd = (node_t*)sys_malloc(sizeof(node_t));
    if(nd == NULL) {
        return NULL;
    }
    node_init(nd, op_ty, 0);
    nd->opset = 0;
    nd->graph = NULL;
    nd->mdl = NULL;
    nd->node_proto = NULL;
    nd->op = device_find_op(internal_device_find("cpu"), nd->op->type);
    if(name) {
        nd->name = sys_strdup(name);
    }
    return nd;
}

node_t * node_new(graph_t* g, const char* name, op_type_t op_ty) {
    node_t * nd = (node_t*)sys_malloc(sizeof(node_t));
    if(nd == NULL) {
        return NULL;
    }
    node_init(nd, op_ty, g->nnode);
    nd->opset = 0;
    nd->graph = g;
    nd->mdl = g->mdl;
    nd->node_proto = NULL;
    if(name) {
        nd->name = sys_strdup(name);
    }
    return nd;
}

attribute_t* node_get_attr(node_t *nd, const char *name) {
    attribute_t *attr;
    int i;
    if(nd && name && nd->attr_vec) {
        for(i = 0; i < vector_size(nd->attr_vec); i++) {
            attr = nd->attr_vec[i];
            if(strcmp(attr->name, name) == 0) {
                return attr;
            }
        }
    }
    return NULL;
}

float node_get_attr_float(node_t *nd, const char *name, float dft) {
    attribute_t *attr = node_get_attr(nd, name);
    if(attr && (attr->type == ATTRIBUTE_TYPE_FLOAT))
        return attr->f;
    return dft;
}

int64_t node_get_attr_int(node_t *nd, const char *name, int64_t dft) {
    attribute_t *attr = node_get_attr(nd, name);
    if(attr && (attr->type == ATTRIBUTE_TYPE_INT))
        return attr->i;
    return dft;
}

char * node_get_attr_string(node_t *nd, const char *name, char *dft) {
    attribute_t *attr = node_get_attr(nd, name);
    if(attr && (attr->type == ATTRIBUTE_TYPE_STRING)) {
        if(attr->ns > 0) {
            attr->ss[attr->ns] = 0;
            return attr->ss;
        }
    }
    return dft;
}

int node_get_attr_floats(node_t *nd, const char *name, float ** fs) {
    attribute_t *attr = node_get_attr(nd, name);
    if(attr && (attr->type == ATTRIBUTE_TYPE_FLOATS)) {
        *fs = attr->fs;
        return attr->nf;
    }
    return 0;
}

int node_get_attr_ints(node_t *nd, const char *name, int64_t ** is) {
    attribute_t *attr = node_get_attr(nd, name);
    if(attr && (attr->type == ATTRIBUTE_TYPE_INTS)) {
        *is = attr->is;
        return attr->ni;
    }
    return 0;
}

tensor_t* node_get_attr_tensor(node_t *nd, const char *name, tensor_t * t) {
    attribute_t *attr = node_get_attr(nd, name);
    if(attr && (attr->type == ATTRIBUTE_TYPE_TENSOR)) {
        return attr->t;
    }
    return t;
}

void node_dump(node_t *nd) {
    if(!nd) return;
    int i;
    if(nd) {
        LOG_INFO("%s-%d: %s\r\n", op_name(nd->op->type) ? op_name(nd->op->type) : "Uninit" , nd->opset, nd->name);
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

void node_free(node_t* nd) {
    if(!nd) {
        return;
    }
    if(nd->name) {
        sys_free(nd->name);
        nd->name = NULL;
    }
    if(nd->nin > 0 && nd->in) {
        sys_free(nd->in);
        nd->in= NULL;
        nd->nin = 0;
    }
    if(nd->nout > 0 && nd->out) {
        sys_free(nd->out);
        nd->out = NULL;
        nd->nout = 0;
    }
    if(nd->priv) {
        sys_free(nd->priv);
        nd->priv = NULL;
    }
    if(nd->attr_vec) { vector_free(nd->attr_vec); nd->attr_vec = NULL; }
    sys_free(nd);
    nd = NULL;
}