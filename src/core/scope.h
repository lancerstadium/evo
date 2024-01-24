
#ifndef CORE_SCOPE_H
#define CORE_SCOPE_H

#include "data.h"

// 域
typedef struct scope {
    int flags;
    Vector* entities;
    struct scope* parent;
} Scope;

#endif