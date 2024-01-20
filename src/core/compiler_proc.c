

#include "compiler.h"


CompileProcess* compile_process_create(const char* filename, const char* out_filename, int flags) {
    LOG_TAG
    FIO* in_file = fio_open(filename);
    if (!in_file) {
        return NULL;
    }

    FILE* out_file = NULL;
    if (out_filename) {
        out_file = fopen(out_filename, "w");
        if (!out_file) {
            fio_close(in_file);
            return NULL;
        }
    }
    CompileProcess* cproc = malloc(sizeof(CompileProcess));
    *cproc = (CompileProcess){
        .flags = flags,
        .cfile = in_file,
        .ofile = out_file
    };

    return cproc;
}


void compile_process_free(CompileProcess* process) {
    LOG_TAG
    fio_close(process->cfile);
    if (process->ofile) {
        fclose(process->ofile);
    }
    free(process);
}