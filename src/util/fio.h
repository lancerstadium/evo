


// ==================================================================================== //
//                                   Util: File I/O
// ==================================================================================== //

#ifndef UTIL_FIO_H
#define UTIL_FIO_H

#include "str.h"
#include <stdio.h>


typedef struct {
    char *filename;
    Str *buffer;
    FILE *file;
    u64 filelen;
} FIO;

ALLOC_DEC_TYPE(FIO)


// ==================================================================================== //
//                                    Pub API: FIO
// ==================================================================================== //

FIO *fio_open(const char *filename, const char *mode);
void fio_close(FIO *fio);
void fio_write(FIO *fio, Str *data);
void fio_flush(FIO *fio);


#endif // UTIL_FIO_H