#ifndef UTIL_BUFFER_H
#define UTIL_BUFFER_H

#include "gtype.h"

#define BUFFER_REALLOC_AMOUNT 2000
typedef struct buffer
{
    char* data;
    // Read index
    int rindex;
    int len;
    int msize;
} Buffer;

Buffer* buffer_create();

char buffer_read(Buffer* buffer);
char buffer_peek(Buffer* buffer);

void buffer_extend(Buffer* buffer, size_t size);
void buffer_printf(Buffer* buffer, const char* fmt, ...);
void buffer_printf_append(Buffer* buffer, const char* fmt, ...);
void buffer_write(Buffer* buffer, char c);
void* buffer_ptr(Buffer* buffer);
void buffer_free(Buffer* buffer);


#endif