
#include "fio.h"
#include "str.h"
#include "log.h"


// ==================================================================================== //
//                                    Pub API: FIO
// ==================================================================================== //

// Opens the file
FIO* fio_open(const char* filename) {
    int err = 0;
    int sz = 0;
    Vector* vec = NULL;
    FILE* f = fopen(filename, "r");
    FIO *fio = NULL;
    if (!f) {
        err = -1;
        goto out;
    }
    fseek(f, 0L, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    vec = vector_create(sizeof(char));
    if(vector_fread(vec, sz, f) < 0)
    {
        err = -1;
        goto out;
    }
    fseek(f, 0, SEEK_SET);
    fio = malloc(sizeof(FIO));
    if(!fio) {
        err = -1;
        goto out;
    }
    char* in_path = (char*)malloc(strlen(filename) + 1);
    if(!in_path) {
        err = -1;
        goto out;
    }
    strcpy(in_path, filename);
    *fio = (FIO){
        .fp = f,
        .size = sz,
        .vec = vec,
        .path = in_path
    };

out:
    if(err < 0) {
        log_error("Open file: %s failed!", filename);
        if(vec) {
            vector_free(vec);
        }
        if(fio) {
            free(fio);
        }
    }
    return fio;
}

// Closes the file
void fio_close(FIO* fio) {
    if(!fio) return;
    free(fio->path);
    fclose(fio->fp);
    vector_free(fio->vec);
    fio = NULL;
}
// Pops the next character
char fio_peek(FIO* fio) {
    char* c = (char*)vector_peek(fio->vec);
    vector_pop(fio->vec);
    return *c;
}
// Reads until the given string is found
Vector* fio_read_until(FIO* fio, const char* delims) {
    Vector* vec = vector_create(sizeof(char));
    int totoal_delims = strlen(delims);
    char c = fio_peek(fio);
    while(!char_is_delim(c, delims)) {
        vector_push(vec, &c);
    }
    return vec;
}

char* fio_get_bare_filename(FIO* fio) {
    char* bare_fname = (char*)malloc(strlen(fio->path) + 1);
    strcpy(bare_fname, fio->path);
    char *dot = strrchr(bare_fname, '.'); // 寻找最后一个点的位置
    
    if (dot != NULL) {
        *dot = '\0'; // 将最后一个点及其后的部分置为字符串结束符
    }
    return bare_fname;
}