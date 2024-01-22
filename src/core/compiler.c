

#include "compiler.h"
#include "lexer.h"
#include "parser.h"


int compile_file(const char* filename, const char* out_filename, int flags) {

    LOG_TAG
    // Step0: Initial
    CompileProcess* cproc = compile_process_create(filename, out_filename, flags);
    if (!cproc) {
        return COMPILER_FILE_ERROR;
    }
    LexProcess* lproc = lex_process_create(cproc, NULL);
    if (!lproc) {
        compile_process_free(cproc);
        return COMPILER_FILE_ERROR;
    }
    ParseProcess* pproc = parse_process_create(lproc);
    if (!pproc) {
        lex_process_free(lproc);
        return COMPILER_FILE_ERROR;
    }

    // Step1: Perform Tokenizing
    if(lex(lproc) != LEXER_ANALYSIS_OK) {
        lex_process_free(lproc);
        compile_process_free(cproc);
        return COMPILER_FILE_ERROR;
    }

    /// Step2: Perform Parsing
    if(parse(pproc) != PARSER_ANALYSIS_OK) {
        parse_process_free(pproc);
        lex_process_free(lproc);
        compile_process_free(cproc);
        return COMPILER_FILE_ERROR;
    }

    /// TODO: Perform Code Generation


    // Step-1: Free memory
    parse_process_free(pproc);
    lex_process_free(lproc);
    compile_process_free(cproc);
    return COMPILER_FILE_OK;
}


