
#include "codegen.h"

#define asm_push(fmt, ...) buffer_printf(cgproc->asm_code, fmt "\n", __VA_ARGS__)

// 删除buffer中最后一行字符串
#define asm_pop() \
    do {\
        int start_pos = cgproc->asm_code->len-1;\
        int newline_pos = start_pos-1;\
        while(newline_pos >= 0 && cgproc->asm_code->data[newline_pos] != '\n') {\
            newline_pos--;\
        }\
        if(newline_pos >= 0) {\
            cgproc->asm_code->len = newline_pos;\
            cgproc->asm_code->data[cgproc->asm_code->len] = '\0';\
        } \
    } while(0)



Node* codegen_process_peek_node(CodegenProcess* cgproc) {
    Node* nd = vector_peek_no_increment(cgproc->parse_proc->node_vec);
    if(!nd) {
        return NULL;
    }
    return nd;
}

Node* codegen_process_next_node(CodegenProcess* cgproc) {
    Node* nd = vector_peek(cgproc->parse_proc->node_vec);
    if(!nd) {
        return NULL;
    }
    return nd;
}

CodegenProcess* codegen_process_create(ParseProcess* pproc) {
    LOG_TAG
    CodegenProcess* cgproc = malloc(sizeof(CodegenProcess));
    *cgproc = (CodegenProcess) {
        .asm_code = buffer_create(),
        .parse_proc = pproc,
        .peek_node = codegen_process_peek_node,
        .next_node = codegen_process_next_node
    };

    buffer_printf(cgproc->asm_code, "\n");
    return cgproc;
}

void codegen_process_free(CodegenProcess* cgproc) {
    LOG_TAG
    if(!cgproc) {
        return;
    }
    buffer_free(cgproc->asm_code);
    if(cgproc->next_node) cgproc->next_node = NULL;
    free(cgproc);
}

int codegen_process_next(CodegenProcess* cgproc) {
    Node* nd = cgproc->next_node(cgproc);
    int res = 0;
    if(!nd) {
        return -1;
    }
    switch(nd->type) {
        case NODE_TYPE_MOD:
            asm_push(".globl %s", nd->mod.name);
            asm_push("%s:", nd->mod.name);
            break;
        case NODE_TYPE_NUM:
            asm_push("  movl $%d, %%eax", nd->llnum);
            break;
        case NODE_TYPE_EOF:
            res = -1;
            break;
        case NODE_TYPE_PROG:
        default: break;
    }
    return res;
}

void codegen_process_root(CodegenProcess* cgproc) {
    Node* root = cgproc->parse_proc->root;
    if(!root) {
        return;
    }else if(root->type != NODE_TYPE_PROG) {
        codegen_error("root must be prog node");
        return;
    }
    while(codegen_process_next(cgproc) == 0) {
        
    }
}
