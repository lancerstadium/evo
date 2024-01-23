
#include "node.h"


void node_read(Node* nd) {
    if(!nd) return;
    Buffer* buf = buffer_create();
    buffer_printf(buf, "\n Read Node: \n");
    switch(nd->type) {
        case NODE_TYPE_EXPRESSION:
            buffer_printf(buf, "   type          : expression\n");
            buffer_printf(buf, "   expr op       : %s\n", nd->expr.op);
            break;
        case NODE_TYPE_FUNCTION:
            buffer_printf(buf, "   type          : function\n");
            buffer_printf(buf, "   func datatype : %s\n", nd->func.type->type_str);
            break;
        case NODE_TYPE_NUMBER:
            buffer_printf(buf, "   type          : number\n");
            buffer_printf(buf, "   number data   : %lld\n", nd->llnum);
            break;
        case NODE_TYPE_VARIABLE:
            buffer_printf(buf, "   type          : variable\n");
            buffer_printf(buf, "   var name      : %s\n", nd->var.name);
            break;
        case NODE_TYPE_IDENTIFIER:
            buffer_printf(buf, "   type          : identifier\n");
            buffer_printf(buf, "   ident name    : %s\n", nd->sval);
            break;
        default:
            break;
    }
    log_info(buffer_ptr(buf));
}