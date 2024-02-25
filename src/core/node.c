
#include "node.h"


static const char* node_type_str[] = {
    [NODE_TYPE_PROG]   = "prog",
    [NODE_TYPE_MOD]    = "mod",
    [NODE_TYPE_NUM]    = "number",
    [NODE_TYPE_IDENT]  = "ident",
    [NODE_TYPE_EXPR]   = "expr",
    [NODE_TYPE_PARAM]  = "param",
    [NODE_TYPE_FUNC]   = "func",
    [NODE_TYPE_BODY]   = "body",
    [NODE_TYPE_STMT]   = "stmt",
    [NODE_TYPE_STRUCT] = "struct",
    [NODE_TYPE_ENUM]   = "enum",
    [NODE_TYPE_EOF]    = "EOF",
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

void node_write_buffer(Node* nd, Buffer* buf) {
    if(!nd) return;
    if(buf->len == 0) {
        buffer_printf(buf, "\n");
    }
    for(int i = 0; i < nd->depth; i++) {
        buffer_printf(buf, "  ");
    }
    buffer_printf(buf, "- %-6s : ", node_get_type_str(nd));
    switch(nd->type) {
        case NODE_TYPE_PROG:
            buffer_printf(buf, "%s\n", nd->prog.name); break;
        case NODE_TYPE_MOD:
            buffer_printf(buf, "%s\n", nd->mod.name); break;
        case NODE_TYPE_FUNC:
            buffer_printf(buf, "%s\n", nd->func.name); break;
        case NODE_TYPE_IDENT:
            buffer_printf(buf, "%s %s\n", nd->sval, nd->ident.dtype.type_str); break;
        case NODE_TYPE_STRUCT:
            buffer_printf(buf, "%s\n", nd->stc.name); break;
        case NODE_TYPE_ENUM:
            buffer_printf(buf, "%s\n", nd->enm.name); break;
        case NODE_TYPE_STMT:
            buffer_printf(buf, "%s\n", nd->sval); break;
        case NODE_TYPE_BODY:
            buffer_printf(buf, "(");
            if(nd->body.sym_tbl) {
                for(char* key = (char*) hashmap_first(nd->body.sym_tbl); key != NULL; key = (char*) hashmap_next(nd->body.sym_tbl, key)) {
                    Node* id_nd = hashmap_get(nd->body.sym_tbl, key);
                    if(!STR_EQ(key, hashmap_first(nd->body.sym_tbl))) {
                        buffer_printf(buf, ", ");
                    }
                    buffer_printf(buf, "%s %s", id_nd->sval, id_nd->ident.dtype.type_str);
                }
            }
            buffer_printf(buf, ")\n");
            break;
        case NODE_TYPE_PARAM:
            buffer_printf(buf, "(");
            for(char* key = (char*) hashmap_first(nd->param.sym_tbl); key != NULL; key = (char*) hashmap_next(nd->param.sym_tbl, key)) {
                Node* id_nd = hashmap_get(nd->param.sym_tbl, key);
                if(!STR_EQ(key, hashmap_first(nd->param.sym_tbl))) {
                    buffer_printf(buf, ", ");
                }
                buffer_printf(buf, "%s %s", id_nd->sval, id_nd->ident.dtype.type_str);
            }
            buffer_printf(buf, ")\n");
            break;
        case NODE_TYPE_EXPR:
            buffer_printf(buf, "%s\n", nd->expr.op); break;
        case NODE_TYPE_NUM:
            buffer_printf(buf, "%d\n", nd->llnum); break;
        case NODE_TYPE_EOF: break;
        default:
            buffer_printf(buf, "\n"); break;
    }
}

void node_read(Node* nd) {
    if(!nd) return;
    Buffer* buf = buffer_create();
    buffer_printf(buf, "\n Read Node: \n");
    buffer_printf(buf, "   type (depth)  : %-10s (%d)\n", node_get_type_str(nd), nd->depth);
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
            buffer_printf(buf, "   func return   : %s", datatype_str[nd->func.fn_rtype.type]);
            switch (nd->func.fn_rtype.type) {
                default: buffer_printf(buf, " (None)");
            }
            buffer_printf(buf, "\n");
            break;
        case NODE_TYPE_NUM:
            buffer_printf(buf, "   number data   : %lld\n", nd->llnum);
            break;
        case NODE_TYPE_VAR:
            buffer_printf(buf, "   var name      : %s\n", nd->var.name);
            break;
        case NODE_TYPE_IDENT:
            buffer_printf(buf, "   ident name    : %s\n", nd->sval);
            buffer_printf(buf, "   ident pos     : %s:%d:%d\n", nd->ident.pos->filename, nd->ident.pos->line, nd->ident.pos->col);
            buffer_printf(buf, "   ident type    : %s\n", nd->ident.dtype.type_str ? nd->ident.dtype.type_str : "(None)");
            break;
        case NODE_TYPE_BODY:
            break;
        case NODE_TYPE_ENUM:
            buffer_printf(buf, "   enum name     : %s\n", nd->enm.name);
            break;
        case NODE_TYPE_STRUCT:
            buffer_printf(buf, "   struct name   : %s\n", nd->stc.name);
            break;
        case NODE_TYPE_PARAM:
            buffer_printf(buf, "   belong to     : %s\n", nd->pnd->func.name);
            break;
        default:
            break;
    }
    log_debug(buffer_ptr(buf));
}