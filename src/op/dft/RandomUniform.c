#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>

typedef struct {
    tensor_type_t dtype;
    float high;
    float low;
    float seed;
    int* shape;
    int nshape;
} operator_pdata_t;

static void RandomUniform_operator(node_t* nd) {
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* y = nd->out[0];

    if (pdat->seed != 0.0)
        srand(pdat->seed);
    switch (pdat->dtype) {
        case TENSOR_TYPE_FLOAT16: {
            uint16_t* py = (uint16_t*)y->datas;
            for (size_t i = 0, l = y->ndata; i < l; i++)
                py[i] = float16_to_float32(((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float* py = (float*)y->datas;
            for (size_t i = 0, l = y->ndata; i < l; i++)
                py[i] = ((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double* py = (double*)y->datas;
            for (size_t i = 0, l = y->ndata; i < l; i++)
                py[i] = ((double)rand() / (double)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
        } break;
        default:
            break;
    }
}

void op_RandomUniform_dft(node_t* nd) {
    // 1. RandomUniform init
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nout == 1) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    int64_t* ints;
    int i;
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->nshape = node_get_attr_ints(nd, "shape", &ints);
        if ((pdat->nshape > 0) && (pdat->shape = malloc(sizeof(int) * pdat->nshape))) {
            pdat->dtype = (tensor_type_t)node_get_attr_int(nd, "dtype", 0);
            pdat->high = node_get_attr_float(nd, "high", 1.0);
            pdat->low = node_get_attr_float(nd, "low", 0.0);
            pdat->seed = node_get_attr_float(nd, "seed", 0.0);
            for (i = 0; i < pdat->nshape; i++)
                pdat->shape[i] = ints[i];
            nd->priv = pdat;
        } else {
            free(pdat);
            return;
        }
    }
    // 2. RandomUniform reshape
    tensor_t* y = nd->out[0];
    y->type = pdat->dtype;
    tensor_reshape(y, pdat->nshape, pdat->shape);
    // 3. RandomUniform run
    RandomUniform_operator(nd);
    // 4. RandomUniform exit
    if (pdat) {
        if (pdat->shape)
            free(pdat->shape);
        free(pdat);
    }
    nd->priv = NULL;
    return;
}