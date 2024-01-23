

#ifndef CORE_CODEGEN_H
#define CORE_CODEGEN_H

#include "parser.h"

#define codegen_error(...) log_error( _bmag("[Codegen]") " " __VA_ARGS__)

typedef enum {
    CODE_GENERATOR_OK,
    CODE_GENERATOR_ERROR
} CodeGeneratorResult;

typedef struct codegen_process CodegenProcess;
typedef Node* (*CODEGEN_PROCESS_NEXT_ND)(CodegenProcess* cgproc);

struct codegen_process {

    Buffer* asm_code;                       // 存储汇编代码
    ParseProcess* parse_proc;               // 语法分析器进程

    struct states {
        Vector* expr;
    } states;

    CODEGEN_PROCESS_NEXT_ND next_node;
};


// codegen.c
int codegen(CodegenProcess* cgproc);

// codegen_proc.c
CodegenProcess* codegen_process_create(ParseProcess* pproc);
void codegen_process_free(CodegenProcess* cgproc);
void codegen_process_root(CodegenProcess* cgproc);

#endif