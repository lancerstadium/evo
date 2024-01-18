

#include "evm.h"

Evm* evm_new() {
    Evm *vm = malloc(sizeof(Evm));
    evm_init(vm);
    return vm;
}
void evm_init(Evm *vm) {
    vm->stack = EValue_alloc_new();
}