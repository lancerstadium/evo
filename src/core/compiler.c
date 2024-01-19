

#include "compiler.h"
#include "lexer.h"


int compile_file(const char* filename, const char* out_filename, int flags) {

    LOG_TAG
    CompileProcess* cproc = compile_process_create(filename, out_filename, flags);
    if (!cproc) {
        return COMPILER_FILE_ERROR;
    }
    LexProcess* lproc = lex_process_create(cproc, NULL);
    if (!lproc) {
        compile_process_free(cproc);
        return COMPILER_FILE_ERROR;
    }

    // 解析
    if(lex(lproc) != LEXER_ANALYSIS_OK) {
        lex_process_free(lproc);
        compile_process_free(cproc);
        return COMPILER_FILE_ERROR;
    }

    /// TODO: Perform Parsing

    /// TODO: Perform Code Generation

    lex_process_free(lproc);
    compile_process_free(cproc);
    return COMPILER_FILE_OK;
}


