#include <evo/resolver.h>
#include <evo/util/math.h>
#include <math.h>

typedef struct {
    float epsilon;
    float momentum;
} operator_pdata_t;

static void BatchNormalization_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *scale = nd->in[1];
    tensor_t *b = nd->in[2];
    tensor_t *mean = nd->in[3];
    tensor_t *var = nd->in[4];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    uint16_t *pscale = (uint16_t *)scale->datas;
    uint16_t *pb = (uint16_t *)b->datas;
    uint16_t *pmean = (uint16_t *)mean->datas;
    uint16_t *pvar = (uint16_t *)var->datas;
    uint16_t *py = (uint16_t *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;
        for (i = 0; i < channel; i++)
            py[o + i] = float32_to_float16(float16_to_float32(pscale[jc]) * ((float16_to_float32(px[o + i]) - float16_to_float32(pmean[jc])) / sqrtf(float16_to_float32(pvar[jc]) + pdat->epsilon)) + float16_to_float32(pb[jc]));
    }
}

static void BatchNormalization_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *scale = nd->in[1];
    tensor_t *b = nd->in[2];
    tensor_t *mean = nd->in[3];
    tensor_t *var = nd->in[4];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    float *pscale = (float *)scale->datas;
    float *pb = (float *)b->datas;
    float *pmean = (float *)mean->datas;
    float *pvar = (float *)var->datas;
    float *py = (float *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;
        for (i = 0; i < channel; i++)
            py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / sqrtf(pvar[jc] + pdat->epsilon)) + pb[jc];
    }
}

static void BatchNormalization_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *scale = nd->in[1];
    tensor_t *b = nd->in[2];
    tensor_t *mean = nd->in[3];
    tensor_t *var = nd->in[4];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    double *pscale = (double *)scale->datas;
    double *pb = (double *)b->datas;
    double *pmean = (double *)mean->datas;
    double *pvar = (double *)var->datas;
    double *py = (double *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;
        for (i = 0; i < channel; i++)
            py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / sqrt(pvar[jc] + pdat->epsilon)) + pb[jc];
    }
}

void BatchNormalization_init(node_t *nd) {
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin == 5) || !(nd->nout >= 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->epsilon = node_get_attr_float(nd, "epsilon", 1e-5);
        pdat->momentum = node_get_attr_float(nd, "momentum", 0.9);
        nd->priv = pdat;
    }
}

void BatchNormalization_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, x->type);
}

void BatchNormalization_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_FLOAT16:
            BatchNormalization_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            BatchNormalization_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            BatchNormalization_forward_float64(nd);
            break;
        default:
            break;
    }
}

void BatchNormalization_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_BatchNormalization_dft(node_t *nd) {
    // 1. BatchNormalization init
    BatchNormalization_init(nd);
    // 2. BatchNormalization reshape
    BatchNormalization_reshape(nd);
    // 3. BatchNormalization run
    BatchNormalization_forward(nd);
    // 4. BatchNormalization exit
    BatchNormalization_exit(nd);
}