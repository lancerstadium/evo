
#include "def.h"

void op_Reshape_dft(node_t* nd) {
    if(!nd || !nd->input_tensors  || nd->input_tensors[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    // Reshape init

    // Reshape run

    // Reshape exit
}