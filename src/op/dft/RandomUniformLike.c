#include <evo/resolver.h>
#include <evo/util/math.h>
#include <string.h>

typedef struct {
    tensor_type_t dtype;
    float high;
    float low;
    float seed;
} operator_pdata_t;


void RandomUniformLike_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        pdat->dtype = (tensor_type_t)node_get_attr_int(nd, "dtype", 0);
        pdat->high = node_get_attr_float(nd, "high", 1.0);
        pdat->low = node_get_attr_float(nd, "low", 0.0);
        pdat->seed = node_get_attr_float(nd, "seed", 0.0);
        nd->priv = pdat;
    }
}

void RandomUniformLike_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    tensor_type_t type;
    if (pdat->dtype != TENSOR_TYPE_UNDEFINED)
        type = pdat->dtype;
    else
        type = x->type;
    switch (type) {
        case TENSOR_TYPE_FLOAT16:
        case TENSOR_TYPE_FLOAT32:
        case TENSOR_TYPE_FLOAT64:
            y->type = type;
            tensor_reshape(y, x->ndim, x->dims);
            break;
        default:
            break;
    }
}

void RandomUniformLike_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
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

void RandomUniformLike_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_RandomUniformLike_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = RandomUniformLike_init;
    nd->op->reshape     = RandomUniformLike_reshape;
    nd->op->forward     = RandomUniformLike_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = RandomUniformLike_exit;
}