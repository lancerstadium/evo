#include <evo/resolver.h>
#include <evo/util/log.h>


void op_Nop_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = NULL;
    nd->op->reshape     = NULL;
    nd->op->forward     = NULL;
    nd->op->backward    = NULL;
    nd->op->exit        = NULL;
}