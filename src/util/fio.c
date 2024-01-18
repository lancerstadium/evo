

#include "fio.h"


ALLOC_DEF_METHOD(FIO)

// ==================================================================================== //
//                                    Pub API: FIO
// ==================================================================================== //

FIO *fio_open(const char *filename, const char *mode) {
    FIO *fio = malloc(sizeof(FIO));
    memcpy(fio->filename, filename, strlen(filename));
    fio->buffer = str_new("");
    fio->file = fopen(filename, mode);
    if(!fio->file) {
        free(fio->buffer);
        free(fio);
        return NULL;
    }
    // 求文件长度
    fseek(fio->file, 0, SEEK_END);
    fio->filelen = ftell(fio->file);
    fseek(fio->file, 0, SEEK_SET);
    return fio;
}
void fio_close(FIO *fio) {
    fclose(fio->file);
    str_free(fio->buffer);
    free(fio);
}
void fio_write(FIO *fio, Str *data) {
    fwrite(data->s, sizeof(char), data->len, fio->file);
}
void fio_flush(FIO *fio) {
    fflush(fio->file);
}