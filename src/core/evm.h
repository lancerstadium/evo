

#ifndef CORE_EVM_H
#define CORE_EVM_H

#include "etype.h"


typedef struct Evm {
    EValue_alloc_t *stack;
    /// TODO:
} Evm;

Evm* evm_new();
void evm_init(Evm *vm);



#endif // CORE_EVM_H