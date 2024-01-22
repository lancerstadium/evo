

#ifndef UTIL_POS_H
#define UTIL_POS_H

#include "log.h"


#define POS_INIT  ((struct pos){__LINE__, 0, __FILE__})

typedef struct pos {
    int line;
    int col;
    const char* filename;
} Pos;

void pos_free(Pos* pos);


#endif // UTIL_POS_H