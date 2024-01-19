


// ==================================================================================== //
//                                   Util: File I/O
// ==================================================================================== //

#ifndef UTIL_FIO_H
#define UTIL_FIO_H

#include "vector.h"

// ==================================================================================== //
//                                    Pub API: FIO
// ==================================================================================== //

typedef struct {
    // The file pointer
    FILE* fp;
    // The file path
    const char* path;
    // The file size
    size_t size;
    // Vector of characters in the file.
    Vector* vec;
} FIO;

// Opens the file
FIO* fio_open(const char* filename);
// Closes the file
void fio_close(FIO* fio);
// Pops the next character
char fio_peek(FIO* fio);
// Reads until the given string is found
Vector* fio_read_until(FIO* fio, const char* delims);


#endif // UTIL_FIO_H