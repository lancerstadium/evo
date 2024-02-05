
#include "node.h"


void node_swap(Node** nd1, Node** nd2) {
    Node* tmp_nd = *nd1;
    *nd1 = *nd2;
    *nd2 = tmp_nd;
}

void node_append_size(Node* nd, size_t *var_size) {
    if(nd->type == NODE_TYPE_VAR) {
        *var_size += nd->var.type->size;
    }
}

void node_read(Node* nd) {
    if(!nd) return;
    Buffer* buf = buffer_create();
    buffer_printf(buf, "\n Read Node: \n");
    switch(nd->type) {
        case NODE_TYPE_EXPR:
            buffer_printf(buf, "   type          : expression\n");
            buffer_printf(buf, "   expr op       : %s\n", nd->expr.op);
            break;
        case NODE_TYPE_FUNC:
            buffer_printf(buf, "   type          : function\n");
            buffer_printf(buf, "   func return   : %s\n", nd->func.rtype->type_str);
            break;
        case NODE_TYPE_NUM:
            buffer_printf(buf, "   type          : number\n");
            buffer_printf(buf, "   number data   : %lld\n", nd->llnum);
            break;
        case NODE_TYPE_VAR:
            buffer_printf(buf, "   type          : variable\n");
            buffer_printf(buf, "   var name      : %s\n", nd->var.name);
            break;
        case NODE_TYPE_IDENT:
            buffer_printf(buf, "   type          : identifier\n");
            buffer_printf(buf, "   ident name    : %s\n", nd->sval);
            break;
        case NODE_TYPE_BODY:
            buffer_printf(buf, "   type          : body\n");
            // buffer_printf(buf, "   ident name    : %s\n");
        default:
            break;
    }
    log_debug(buffer_ptr(buf));
}