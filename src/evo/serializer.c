
#include "evo.h"
#include "onnx.proto3.pb-c.h"
#include "sys.h"
#include <string.h>

// ==================================================================================== //
//                                      onnx
// ==================================================================================== //

context_t* load_onnx(struct serializer* s, const void* buf, int len) {
    context_t* ctx = NULL;
    if (!buf || len <= 0)
        return NULL;
    ctx = context_new(NULL);
    ctx->sez = s;
    ctx->model = onnx__model_proto__unpack(NULL, len, buf);
    ctx->model_size = len;
    if (!ctx->model) {
        if (ctx)
            free(ctx);
        return NULL;
    }
    return ctx;
}

context_t* load_file_onnx(struct serializer* sez, const char* path) {
    context_t* ctx = NULL;
    FILE* fp;
    uint32_t len;
    unsigned int i;
    void* buf;
    fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        len = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        if (len > 0) {
            buf = sys_malloc(len);
            if (buf) {
                for (i = 0; i < len; i += fread(buf + i, 1, len - i, fp));
                ctx = load_onnx(sez, buf, len);
                free(buf);
            }
        }
        fclose(fp);
    }
    return ctx;
}

void unload_onnx(context_t* ctx) {
    if (ctx && ctx->model) {
        onnx__model_proto__free_unpacked(ctx->model, NULL);
        ctx->model_size = 0;
    }
}

EVO_UNUSED static tensor_t* tensor_from_value_info(Onnx__ValueInfoProto* v) {
    tensor_t* t;
    EVO_UNUSED tensor_type_t type;
    int* dims = NULL;
    int ndim;
    int i;

    if (!v || !v->name)
        return NULL;

    switch (v->type->value_case) {
        case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
            type = (tensor_type_t)v->type->tensor_type->elem_type;
            ndim = v->type->tensor_type->shape->n_dim;
            if (ndim > 0) {
                dims = sys_malloc(sizeof(int) * ndim);
                if (dims) {
                    for (i = 0; i < ndim; i++) {
                        switch (v->type->tensor_type->shape->dim[i]->value_case) {
                            case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
                                dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
                                break;
                            case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
                                if (strcmp(v->type->tensor_type->shape->dim[i]->dim_param, "batch_size") == 0)
                                    dims[i] = 1;
                                else
                                    dims[i] = 1;
                                break;
                            default:
                                dims[i] = 1;
                                break;
                        }
                    }
                }
            }
            // t = tensor_alloc(v->name, type, dims, ndim);
            if (dims)
                free(dims);
            break;
        case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
            t = NULL;
            break;
        case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
            t = NULL;
            break;
        default:
            t = NULL;
            break;
    }
    return t;
}

// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t* serializer_new() {
    serializer_t* s = (serializer_t*)sys_malloc(sizeof(serializer_t));
    // default load by onnx
    s->load = load_onnx;
    s->load_file = load_file_onnx;
    s->unload = unload_onnx;
    return s;
}

void serializer_free(serializer_t* sez) {
    if (sez) {
        sez->load = NULL;
        sez->load_file = NULL;
        sez->unload = NULL;
        free(sez);
        sez = NULL;
    }
}