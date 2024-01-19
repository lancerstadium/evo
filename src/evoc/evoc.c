

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                  Proc Entry: evoc
// ==================================================================================== //


int main(int argc, char **argv) {

    if (argc == 1) {
        log_warn("usage: %s <file>", argv[0]);
        return 0;
    }

    LOG_TAG
    char* arg = argv[1];
    int res = compile_file(arg, NULL, 0);

    if(res == COMPILER_FILE_OK) {
        log_info("Compile success!");
    }else if(res == COMPILER_FILE_ERROR) {
        log_error("Compile failed!");
    }else {
        log_error("Unknown res code %d", res);
    }

    // if(argc != 2) {
    //     log_error("usage: %s <stmt>", argv[0]);
    //     return 1;
    // }

    // // 解析用户输入
    // Token *tok = evoc_tokenize(argv[1]);
    // Func *prog = evoc_parse(tok);
    // evoc_codegen(prog); 
    return 0;
}