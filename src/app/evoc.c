

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include "evo.h"
#include <time.h>
#include <unistd.h>

// ==================================================================================== //
//                          App: evoc - evo's compile process
// ==================================================================================== //

ap_def_callback(evoc){

    if (argc == 1) {
        log_warn("usage: %s <file>", argv[0]);
        return;
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

}


ap_def_callback(evo_hello) {
    printf("[EVO]: hello, world!\n");
}