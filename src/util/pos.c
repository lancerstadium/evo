
#include "pos.h"


void pos_free(Pos* pos) {
    if(!pos) return;
    free((char*)pos->filename);
    free(pos);
}