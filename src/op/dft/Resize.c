#include <evo/resolver.h>

typedef struct {
    char* coordinate_transformation_mode;
    float cubic_coeff_a;
    int exclude_outside;
    float extrapolation_value;
    char* mode;
    char* nearest_mode;
} operator_pdata_t;

void op_Resize_dft(node_t* nd) {

}