
#include "codegen.h"

static Buffer* tmp_asm_code; 

void asm_push(const char *ins, ...) {
    va_list args;
    va_start(args, ins);
    buffer_printf(tmp_asm_code, ins, args);
    buffer_printf(tmp_asm_code, "\n");
    va_end(args);
}

Node* codegen_process_next_node(CodegenProcess* cgproc) {
    Node** p_nd = vector_peek(cgproc->parse_proc->node_tree_vec);
    if(!p_nd) {
        return NULL;
    }
    return *p_nd;
}

CodegenProcess* codegen_process_create(ParseProcess* pproc) {
    LOG_TAG
    CodegenProcess* cgproc = malloc(sizeof(CodegenProcess));
    tmp_asm_code = buffer_create();
    *cgproc = (CodegenProcess) {
        .asm_code = tmp_asm_code,
        .parse_proc = pproc,
        .next_node = codegen_process_next_node
    };
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


