#include <evo/resolver.h>
#include <string.h>

union onnx_scalar_t {
    uint8_t v_bool;
    int8_t v_int8;
    int16_t v_int16;
    int32_t v_int32;
    int64_t v_int64;
    uint8_t v_uint8;
    uint16_t v_uint16;
    uint32_t v_uint32;
    uint64_t v_uint64;
    uint16_t v_bfloat16;
    uint16_t v_float16;
    float v_float32;
    double v_float64;
    struct {
        float real;
        float imaginary;
    } v_complex64;
    struct {
        double real;
        double imaginary;
    } v_complex128;
};

typedef struct {
    tensor_type_t type;
    union onnx_scalar_t scalar;
    int size;
} operator_pdata_t;

void ConstantOfShape_init(node_t* nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    attribute_t* attr;
    tensor_t* t = NULL;
    int i;
    if (pdat) {
        memset(pdat, 0, sizeof(operator_pdata_t));
        for (i = 0; i < vector_size(nd->attr_vec); i++) {
            attr = nd->attr_vec[i];
            if ((attr->type == ATTRIBUTE_TYPE_TENSOR) && (strcmp(attr->name, "value") == 0)) {
                t = attr->t;
            }
        }
        if (t) {
            pdat->type = (tensor_type_t)t->type;
            switch (t->type) {
                case TENSOR_TYPE_FLOAT32:
                    pdat->scalar.v_float32 = ((float*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_UINT8:
                    pdat->scalar.v_uint8 = ((uint8_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_INT8:
                    pdat->scalar.v_int8 = ((int8_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_UINT16:
                    pdat->scalar.v_uint16 = ((uint16_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_INT16:
                    pdat->scalar.v_int16 = ((int16_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_INT32:
                    pdat->scalar.v_int32 = ((int32_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_BOOL:
                    pdat->scalar.v_bool = ((int8_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_FLOAT16:
                    pdat->scalar.v_float16 = ((uint16_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_BFLOAT16:
                    pdat->scalar.v_bfloat16 = ((uint16_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_INT64:
                    pdat->scalar.v_int64 = ((int64_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_FLOAT64:
                    pdat->scalar.v_float64 = ((double*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_UINT32:
                    pdat->scalar.v_uint32 = ((uint32_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_UINT64:
                    pdat->scalar.v_uint64 = ((uint64_t*)(t->datas))[0];
                    break;
                case TENSOR_TYPE_COMPLEX64:
                    pdat->scalar.v_complex64.real = ((float*)(t->datas))[0];
                    pdat->scalar.v_complex64.imaginary = ((float*)(t->datas))[1];
                    break;
                case TENSOR_TYPE_COMPLEX128:
                    pdat->scalar.v_complex128.real = ((double*)(t->datas))[0];
                    pdat->scalar.v_complex128.imaginary = ((double*)(t->datas))[1];
                    break;
                default:
                    memset(&pdat->scalar, 0, sizeof(union onnx_scalar_t));
                    break;
            }
        } else {
            pdat->type = TENSOR_TYPE_FLOAT32;
            memset(&pdat->scalar, 0, sizeof(union onnx_scalar_t));
        }
        pdat->size = tensor_type_sizeof(pdat->type);
        nd->priv = pdat;
    }
}

void ConstantOfShape_reshape(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
}

void ConstantOfShape_forward(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t* pdat = (operator_pdata_t*)nd->priv;
    tensor_t* x = nd->in[0];
    tensor_t* y = nd->out[0];
    char* p;
    size_t i, l;

    if (x->ndata > 0) {
        int dims[x->ndata];
        for (i = 0; i < x->ndata; i++)
            dims[i] = ((int64_t*)x->datas)[i];
        tensor_reinit(y, pdat->type, x->ndata, dims);
        
    } else {
        tensor_reinit(y, pdat->type, 0, NULL);
    }
    for (i = 0, l = y->ndata, p = y->datas; i < l; i++, p += pdat->size)
        memcpy(p, &pdat->scalar, pdat->size);
}

void ConstantOfShape_exit(node_t* nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t* pdat = malloc(sizeof(operator_pdata_t));
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_ConstantOfShape_dft(node_t* nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = ConstantOfShape_init;
    nd->op->reshape     = ConstantOfShape_reshape;
    nd->op->forward     = ConstantOfShape_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = ConstantOfShape_exit;
}