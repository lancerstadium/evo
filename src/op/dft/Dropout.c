#include <evo/resolver.h>
#include <evo/util/math.h>

static void Dropout_forward_bfloat16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float ratio = 0.0f;
    bool is_train = false;
    if(nd->nin >= 3) {
        tensor_t* r = nd->in[1];
        tensor_t* m = nd->in[2];
        ratio = ((float*)(r->datas))[0];
        is_train = ((bool*)(m->datas))[0];
    }
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if(is_train) {
            py[i] = (rand() % 2 == 0) ? 0 : float32_to_bfloat16(bfloat16_to_float32(px[i]) / (1 - ratio));
        } else {
            py[i] = px[i];
        }
    }
}

static void Dropout_forward_float16(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float ratio = 0.0f;
    bool is_train = false;
    if(nd->nin >= 3) {
        tensor_t* r = nd->in[1];
        tensor_t* m = nd->in[2];
        ratio = ((float*)(r->datas))[0];
        is_train = ((bool*)(m->datas))[0];
    }
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if(is_train) {
            py[i] = (rand() % 2 == 0) ? 0 : float32_to_float16(float16_to_float32(px[i]) / (1 - ratio));
        } else {
            py[i] = px[i];
        }
    }
}

static void Dropout_forward_float32(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *py = (float *)y->datas;
    float ratio = 0.0f;
    bool is_train = false;
    if(nd->nin >= 3) {
        tensor_t* r = nd->in[1];
        tensor_t* m = nd->in[2];
        ratio = ((float*)(r->datas))[0];
        is_train = ((bool*)(m->datas))[0];
    }
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if(is_train) {
            py[i] = (rand() % 2 == 0) ? 0 : px[i] / (1 - ratio);
        } else {
            py[i] = px[i];
        }
    }
}

static void Dropout_forward_float64(node_t *nd) {
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *py = (double *)y->datas;
    float ratio = 0.0f;
    bool is_train = false;
    if(nd->nin >= 3) {
        tensor_t* r = nd->in[1];
        tensor_t* m = nd->in[2];
        ratio = ((float*)(r->datas))[0];
        is_train = ((bool*)(m->datas))[0];
    }
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        if(is_train) {
            py[i] = (rand() % 2 == 0) ? 0 : px[i] / (1 - ratio);
        } else {
            py[i] = px[i];
        }
    }
}

void Dropout_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
}

void Dropout_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout >= 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
    bool is_train = false;
    if(nd->nin >= 3) {
        tensor_t* m = nd->in[2];
        is_train = ((bool*)(m->datas))[0];
        if(is_train && nd->nout >= 2) {
            tensor_t* z = nd->out[1];
            tensor_reshape_ident(z, x, x->type);
        }
    }
}

void Dropout_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin >= 1) || !(nd->nout >= 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BFLOAT16:
            Dropout_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Dropout_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Dropout_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Dropout_forward_float64(nd);
            break;
        default:
            break;
    }
}

void Dropout_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    return;
}

void op_Dropout_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Dropout_init;
    nd->op->reshape     = Dropout_reshape;
    nd->op->forward     = Dropout_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Dropout_exit;
}