#include <evo/resolver.h>
#include <evo/util/math.h>
#include <math.h>

typedef struct {
    float epsilon;
} operator_pdata_t;

static void InstanceNormalization_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *scale = nd->in[1];
    tensor_t *b = nd->in[2];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *pscale = (uint16_t *)scale->datas;
    uint16_t *pb = (uint16_t *)b->datas;
    uint16_t *py = (uint16_t *)y->datas;
    float temp, mean, var;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, l, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        l = o + channel;
        jc = j % C;
        temp = 0;
        for (i = o; i < l; i++)
            temp += float16_to_float32(px[i]);
        mean = temp / channel;
        temp = 0;
        for (i = o; i < l; i++)
            temp += pow(float16_to_float32(px[i]) - mean, 2);
        var = temp / channel;
        for (i = o; i < l; i++)
            py[i] = float32_to_float16(float16_to_float32(pscale[jc]) * ((float16_to_float32(px[i]) - mean) / sqrtf(var + pdat->epsilon)) + float16_to_float32(pb[jc]));
    }
}

static void InstanceNormalization_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *scale = nd->in[1];
    tensor_t *b = nd->in[2];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *pscale = (float *)scale->datas;
    float *pb = (float *)b->datas;
    float *py = (float *)y->datas;
    float temp, mean, var;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, l, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        l = o + channel;
        jc = j % C;
        temp = 0;
        for (i = o; i < l; i++)
            temp += px[i];
        mean = temp / channel;
        temp = 0;
        for (i = o; i < l; i++)
            temp += pow(px[i] - mean, 2);
        var = temp / channel;
        for (i = o; i < l; i++)
            py[i] = pscale[jc] * ((px[i] - mean) / sqrtf(var + pdat->epsilon)) + pb[jc];
    }
}

static void InstanceNormalization_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *scale = nd->in[1];
    tensor_t *b = nd->in[2];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *pscale = (double *)scale->datas;
    double *pb = (double *)b->datas;
    double *py = (double *)y->datas;
    double temp, mean, var;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, l, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        l = o + channel;
        jc = j % C;
        temp = 0;
        for (i = o; i < l; i++)
            temp += px[i];
        mean = temp / channel;
        temp = 0;
        for (i = o; i < l; i++)
            temp += pow(px[i] - mean, 2);
        var = temp / channel;
        for (i = o; i < l; i++)
            py[i] = pscale[jc] * ((px[i] - mean) / sqrt(var + pdat->epsilon)) + pb[jc];
    }
}

void InstanceNormalization_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    if (!(nd->nin == 3) || !(nd->nout >= 1) || (nd->in[0]->ndim == 0) || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = sys_malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->epsilon = node_get_attr_float(nd, "epsilon", 1e-05);
        nd->priv = pdat;
    }
}

void InstanceNormalization_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
}

void InstanceNormalization_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            InstanceNormalization_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            InstanceNormalization_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            InstanceNormalization_forward_float64(nd);
            break;
        default:
            break;
    }
}

void InstanceNormalization_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}


void op_InstanceNormalization_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = InstanceNormalization_init;
    nd->op->reshape     = InstanceNormalization_reshape;
    nd->op->forward     = InstanceNormalization_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = InstanceNormalization_exit;
}