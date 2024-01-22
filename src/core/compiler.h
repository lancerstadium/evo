

#ifndef CORE_COMPILER_H
#define CORE_COMPILER_H

#include "eval.h"

#define compiler_error(...) log_error( _bmag("[Compiler]") " " __VA_ARGS__)


typedef enum {
    COMPILER_FILE_OK,           // 编译成功
    COMPILER_FILE_ERROR         // 编译出错
} CompileProcessStatus;

typedef struct {
    int flags;                  // 编译标志
    FIO* cfile;                 // 编译目标文件
    FILE* ofile;                // 目标储存地址
} CompileProcess;


// compiler.c
int compile_file(const char* filename, const char* out_filename, int flags);

// compile_proc.c
CompileProcess* compile_process_create(const char* filename, const char* out_filename, int flags);
void compile_process_free(CompileProcess* process);



#endif // CORE_COMPILER_H