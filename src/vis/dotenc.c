#include "dotenc.h"

void dot_header(char* buffer, char* name) {
    sprintf(buffer,
        "digraph %s {\n"
        "compound = true\n"
        "ratio = 1.8\n"
        "ranksep = 0.25\n"
        "nodesep = 0.2\n"
        "splines = true\n"
        "graph [rankdir = TB, shape = record, charset = \"UTF-8\", color = cadetblue, fontcolor = black, fontname = \"Consolas\", fontsize = 20, style = \"rounded,dashed,bold\"]\n"
        "node [shape = record, charset = \"UTF-8\", color = black, fontcolor = black, fontname = \"Consolas\", fontsize = 16, style = \"rounded,bold\"]\n"
        "edge [shape = record, charset = \"UTF-8\", color = black, fontname = \"Consolas\", fontsize = 14, style = \"bold\"]\n"
        ,name);
}

void dot_tensor(char* buffer, tensor_t* ts) {
    if(!ts) return;
    char* shape = tensor_dump_shape(ts);
    if(!ts->is_param) {
        sprintf(buffer,
            "%s\nTensor%d [\n"
            "color = brown\n"
            "fontcolor = brown\n"
            "label = \"{%s | %s}\"\n"
            "]\n"
            ,buffer, ts->index, ts->name, shape);
    } else {
        sprintf(buffer,
            "%s\nTensor%d [\n"
            "color = darkslategrey\n"
            "fontcolor = darkslategrey\n"
            "label = \"{%s | %s}\"\n"
            "]\n"
            ,buffer, ts->index, ts->name, shape);
    }

    free(shape);
}

void dot_node(char* buffer, node_t* nd) {
    if(!nd) return;

    size_t nattr = vector_size(nd->attr_vec);
    if(nattr == 0) {
        sprintf(buffer,
            "%s\nNode%d [ label = \"{ %s: %s }\" ]\n"
            ,buffer, nd->index, op_name(nd->op->type), nd->name);
    } else {
        sprintf(buffer,
            "%s\nNode%d [\n"
            "label = \"{ %s: %s | {\n"
            ,buffer, nd->index, op_name(nd->op->type), nd->name);

        for(int i = 0; i < vector_size(nd->attr_vec); i++) {
            char* attr = attribute_dump_value(nd->attr_vec[i]);
            sprintf(buffer,
            "%s%s\\n\n"
            , buffer, attr);
            free(attr);
        }

        sprintf(buffer,
            "%s\n}}\"\n]\n"
            ,buffer);
    }
}

void dot_link_in(char* buffer, int tdx, int ndx) {
    sprintf(buffer,
        "%s\nTensor%d ->  Node%d"
        ,buffer, tdx, ndx);
}

void dot_link_out(char* buffer, int ndx, int tdx) {
    sprintf(buffer,
        "%s\nNode%d ->  Tensor%d"
        ,buffer, ndx, tdx);
}

void dot_subgraph(char* buffer, graph_t* sg) {
    if(!sg || !sg->nodes || !sg->is_sub) return;
    sprintf(buffer,
        "%s\nsubgraph cluster_subgraph%d {\n"
        "label = \"%s\"\n"
        ,buffer, sg->idx, sg->name);

    for(int i = 0; i < sg->nnode; i++) {
        node_t* nd = sg->nodes[i];
        dot_node(buffer, nd);
        for(int j = 0; j < nd->nin; j++) {
            dot_link_in(buffer, nd->in[j]->index, nd->index);
        }
        for(int j = 0; j < nd->nout; j++) {
            dot_link_out(buffer, nd->index, nd->out[j]->index);
        }
        sprintf(buffer,
        "%s\n"
        ,buffer);
    }

    sprintf(buffer,
        "%s\n}\n"
        ,buffer);   
}

void dot_graph(char* buffer, graph_t* g) {
    if(!buffer || !g || g->is_sub) return;
    dot_header(buffer, g->name);
    for(int i = 0; i < g->ntensor; i++) {
        dot_tensor(buffer, g->tensors[i]);
    }
    for(int i = 0; i < vector_size(g->sub_vec); i++) {
        dot_subgraph(buffer, g->sub_vec[i]);
    }
    dot_footer(buffer);
}

void dot_footer(char* buffer) {
    sprintf(buffer,
        "%s\n}\n"
        ,buffer);
}