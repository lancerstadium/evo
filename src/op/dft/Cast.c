#include <evo/resolver.h>
#include <evo/util/math.h>
#include <evo/util/log.h>

#include <string.h>

typedef struct {
    tensor_type_t to;
} operator_pdata_t;

static void Cast_forward_bool(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((px[i] != 0) ? 1.0 : 0.0);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((px[i] != 0) ? 1.0 : 0.0);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1.0 : 0.0;
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1.0 : 0.0;
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%u", (px[i] != 0) ? 1 : 0);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_int8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int8_t *px = (int8_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%d", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_int16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int16_t *px = (int16_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%d", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_int32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int32_t *px = (int32_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%d", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_int64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int64_t *px = (int64_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%ld", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_uint8(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint8_t *px = (uint8_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%u", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_uint16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%u", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_uint32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint32_t *px = (uint32_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%u", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_uint64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint64_t *px = (uint64_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%lu", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_bfloat16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (bfloat16_to_float32(px[i]) != 0.0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((bfloat16_to_float32(px[i])));
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = bfloat16_to_float32(px[i]);
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)(bfloat16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%g", bfloat16_to_float32(px[i]));
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_float16(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    uint16_t *px = (uint16_t *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float16_to_float32(px[i]) != 0.0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float16_to_float32(px[i])));
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float16_to_float32(px[i]);
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)(float16_to_float32(px[i]));
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%g", float16_to_float32(px[i]));
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_float32(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    float *px = (float *)x->datas;
    size_t i, l;
    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0.0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16(px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16(px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%g", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_float64(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    double *px = (double *)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (px[i] != 0.0) ? 1 : 0;
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)px[i];
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)px[i];
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)px[i];
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)px[i];
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)px[i];
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)px[i];
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)px[i];
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)px[i];
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)px[i]);
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)px[i];
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = px[i];
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            char buf[32];
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                sprintf(buf, "%g", px[i]);
                py[i] = strdup(buf);
            }
        } break;
        default:
            break;
    }
}

static void Cast_forward_string(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    char **px = (char **)x->datas;
    size_t i, l;

    switch (pdat->to) {
        case TENSOR_TYPE_BOOL: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)strtoul(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_INT8: {
            int8_t *py = (int8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int8_t)strtol(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_INT16: {
            int16_t *py = (int16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int16_t)strtol(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_INT32: {
            int32_t *py = (int32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int32_t)strtol(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_INT64: {
            int64_t *py = (int64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (int64_t)strtoll(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_UINT8: {
            uint8_t *py = (uint8_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint8_t)strtoul(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_UINT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint16_t)strtoul(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_UINT32: {
            uint32_t *py = (uint32_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint32_t)strtoul(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_UINT64: {
            uint64_t *py = (uint64_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (uint64_t)strtoull(px[i], 0, 0);
        } break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_bfloat16((float)strtod(px[i], NULL));
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *py = (uint16_t *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = float32_to_float16((float)strtod(px[i], NULL));
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *py = (float *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (float)strtod(px[i], NULL);
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *py = (double *)y->datas;
            for (i = 0, l = y->ndata; i < l; i++)
                py[i] = (double)strtod(px[i], NULL);
        } break;
        case TENSOR_TYPE_STRING: {
            char **py = (char **)y->datas;
            for (i = 0, l = y->ndata; i < l; i++) {
                if (py[i])
                    free(py[i]);
                py[i] = strdup(px[i]);
            }
        } break;
        default:
            break;
    }
}

void Cast_init(node_t *nd) {
    if (!nd || !nd->in) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->to = (tensor_type_t)node_get_attr_int(nd, "to", nd->in[0]->type);
        nd->priv = pdat;
    }
}

void Cast_reshape(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    tensor_reshape_ident(y, x, pdat->to);
}

void Cast_forward(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    if (!(nd->nin == 1) || !(nd->nout == 1) 
        || (nd->in[0]->ndim == 0) 
        || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    switch (nd->in[0]->type) {
        case TENSOR_TYPE_BOOL:
            Cast_forward_bool(nd);
            break;
        case TENSOR_TYPE_INT8:
            Cast_forward_int8(nd);
            break;
        case TENSOR_TYPE_INT16:
            Cast_forward_int16(nd);
            break;
        case TENSOR_TYPE_INT32:
            Cast_forward_int32(nd);
            break;
        case TENSOR_TYPE_INT64:
            Cast_forward_int64(nd);
            break;
        case TENSOR_TYPE_UINT8:
            Cast_forward_uint8(nd);
            break;
        case TENSOR_TYPE_UINT16:
            Cast_forward_uint16(nd);
            break;
        case TENSOR_TYPE_UINT32:
            Cast_forward_uint32(nd);
            break;
        case TENSOR_TYPE_UINT64:
            Cast_forward_uint64(nd);
            break;
        case TENSOR_TYPE_BFLOAT16:
            Cast_forward_bfloat16(nd);
            break;
        case TENSOR_TYPE_FLOAT16:
            Cast_forward_float16(nd);
            break;
        case TENSOR_TYPE_FLOAT32:
            Cast_forward_float32(nd);
            break;
        case TENSOR_TYPE_FLOAT64:
            Cast_forward_float64(nd);
            break;
        case TENSOR_TYPE_STRING:
            Cast_forward_string(nd);
            break;
        default:
            break;
    }
}

void Cast_exit(node_t *nd) {
    if(!nd || !nd->in || !nd->out) return;
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    if (pdat)
        free(pdat);
    nd->priv = NULL;
    return;
}

void op_Cast_dft(node_t *nd) {
    if(!nd || !nd->op) return;
    nd->op->init        = Cast_init;
    nd->op->reshape     = Cast_reshape;
    nd->op->forward     = Cast_forward;
    nd->op->backward    = NULL;
    nd->op->exit        = Cast_exit;
}