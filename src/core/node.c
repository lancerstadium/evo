
#include "node.h"


static const char* node_type_str[] = {
    [NODE_TYPE_PROG] = "prog",
    [NODE_TYPE_MOD]  = "mod",
    [NODE_TYPE_EXPR] = "expr",
    [NODE_TYPE_FUNC] = "func",
    [NODE_TYPE_BODY] = "body",
    [NODE_TYPE_EOF]  = "EOF",
};

const char* node_get_type_str(Node* nd){
    return (char*)(node_type_str[nd->type] ? node_type_str[nd->type] : "Unknown");
}

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
    buffer_printf(buf, "   type          : %s\n", node_get_type_str(nd));
    buffer_printf(buf, "   depth         : %d\n", nd->depth);
    buffer_printf(buf, "   addr          : %p\n", nd);
    switch(nd->type) {
        case NODE_TYPE_PROG:
            buffer_printf(buf, "   prog name     : %s\n", nd->prog.name);
            break;
        case NODE_TYPE_MOD:
            buffer_printf(buf, "   mod name      : %s\n", nd->mod.name);
            break;
        case NODE_TYPE_EXPR:
            buffer_printf(buf, "   expr op       : %s\n", nd->expr.op);
            break;
        case NODE_TYPE_FUNC:
            buffer_printf(buf, "   func name     : %s\n", nd->func.name);
            buffer_printf(buf, "   func return   : %d\n", nd->func.rtype->type);
            break;
        case NODE_TYPE_NUM:
            buffer_printf(buf, "   number data   : %lld\n", nd->llnum);
            break;
        case NODE_TYPE_VAR:
            buffer_printf(buf, "   var name      : %s\n", nd->var.name);
            break;
        case NODE_TYPE_IDENT:
            buffer_printf(buf, "   ident name    : %s\n", nd->sval);
            break;
        case NODE_TYPE_BODY:
            break;
        default:
            break;
    }
    log_debug(buffer_ptr(buf));
}