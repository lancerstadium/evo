
#include "codegen.h"

int codegen(CodegenProcess* cgproc) {
    LOG_TAG
    codegen_process_root(cgproc);
    // log_info("%s", buffer_ptr(cgproc->asm_code));
    return CODE_GENERATOR_OK;
}