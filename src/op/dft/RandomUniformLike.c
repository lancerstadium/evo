#include <evo/resolver.h>

typedef struct {
	tensor_type_t dtype;
	float high;
	float low;
	float seed;
} operator_pdata_t;

void op_RandomUniformLike(node_t* nd) {

}