
#include <string.h>

#include "evo.h"
#include "math.h"
#include "onnx.proto3.pb-c.h"
#include "sys.h"

// ==================================================================================== //
//                                      onnx
// ==================================================================================== //

context_t *load_onnx(struct serializer *s, const void *buf, int len) {
    context_t *ctx = NULL;
    if (!buf || len <= 0)
        return NULL;
    ctx = context_new(NULL);
    ctx->sez = s;
    ctx->model = onnx__model_proto__unpack(NULL, len, buf);
    ctx->model_size = len;
    if (!ctx->model) {
        if (ctx)
            sys_free(ctx);
        return NULL;
    }
    return ctx;
}

context_t *load_file_onnx(struct serializer *sez, const char *path) {
    context_t *ctx = NULL;
    FILE *fp;
    uint32_t len;
    unsigned int i;
    void *buf;
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
                sys_free(buf);
            }
        }
        fclose(fp);
    }
    return ctx;
}

void unload_onnx(context_t *ctx) {
    if (ctx && ctx->model) {
        onnx__model_proto__free_unpacked(ctx->model, NULL);
        ctx->model_size = 0;
    }
}

EVO_UNUSED static tensor_t *tensor_from_value_info(Onnx__ValueInfoProto *v) {
    tensor_t *t;
    tensor_type_t type;
    int *dims = NULL;
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
            t = tensor_new(v->name, type);
            tensor_set_shape(t, ndim, dims);
            if (dims)
                sys_free(dims);
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

static void tensor_copy_proto(tensor_t *t, Onnx__TensorProto *o) {
    size_t n, i;
    int sz;

    if (t && o) {
        if (t->type == o->data_type) {
            sz = tensor_type_sizeof(t->type);
            if (sz > 0) {
                if ((o->raw_data.len > 0) && o->raw_data.data) {
                    switch (o->data_type) {
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: {
                            float *p = (float *)t->datas;
                            uint32_t *q = (uint32_t *)o->raw_data.data;
                            union {
                                uint32_t u;
                                float f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++) {
                                    v.u = le32_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: {
                            uint8_t *p = (uint8_t *)t->datas;
                            uint8_t *q = (uint8_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len);
                                memcpy(p, q, n);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: {
                            int8_t *p = (int8_t *)t->datas;
                            int8_t *q = (int8_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len);
                                memcpy(p, q, n);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16: {
                            uint16_t *p = (uint16_t *)t->datas;
                            uint16_t *q = (uint16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT16: {
                            int16_t *p = (int16_t *)t->datas;
                            int16_t *q = (int16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: {
                            int32_t *p = (int32_t *)t->datas;
                            int32_t *q = (int32_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le32_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: {
                            int64_t *p = (int64_t *)t->datas;
                            int64_t *q = (int64_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le64_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL: {
                            uint8_t *p = (uint8_t *)t->datas;
                            uint8_t *q = (uint8_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len);
                                memcpy(p, q, n);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16: {
                            uint16_t *p = (uint16_t *)t->datas;
                            uint16_t *q = (uint16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: {
                            double *p = (double *)t->datas;
                            uint64_t *q = (uint64_t *)o->raw_data.data;
                            union {
                                uint64_t u;
                                double f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++) {
                                    v.u = le64_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32: {
                            uint32_t *p = (uint32_t *)t->datas;
                            uint32_t *q = (uint32_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le32_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64: {
                            uint64_t *p = (uint64_t *)t->datas;
                            uint64_t *q = (uint64_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le64_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64: {
                            float *p = (float *)t->datas;
                            uint32_t *q = (uint32_t *)o->raw_data.data;
                            union {
                                uint32_t u;
                                float f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz) * 2;
                                for (i = 0; i < n; i++) {
                                    v.u = le32_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128: {
                            double *p = (double *)t->datas;
                            uint64_t *q = (uint64_t *)o->raw_data.data;
                            union {
                                uint64_t u;
                                double f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz) * 2;
                                for (i = 0; i < n; i++) {
                                    v.u = le64_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16: {
                            uint16_t *p = (uint16_t *)t->datas;
                            uint16_t *q = (uint16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        default:
                            break;
                    }
                } else {
                    switch (o->data_type) {
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
                            n = MIN(t->ndata, (size_t)o->n_float_data);
                            if ((n > 0) && t->datas && o->float_data)
                                memcpy(t->datas, o->float_data, sizeof(float) * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
                            // TODO
                            n = MIN(t->ndata, (size_t)o->n_int32_data);
                            if ((n > 0) && t->datas && o->int32_data)
                                memcpy(t->datas, o->int32_data, sz * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
                            n = MIN(t->ndata, (size_t)o->n_string_data);
                            if ((n > 0) && t->datas && o->string_data) {
                                char **str = (char **)t->datas;
                                for (i = 0; i < t->ndata; i++) {
                                    if (str[i]) {
                                        sys_free(str[i]);
                                        str[i] = NULL;
                                    }
                                }
                                for (i = 0; i < n; i++) {
                                    str[i] = sys_malloc(o->string_data[i].len + 1);
                                    if (str[i]) {
                                        str[i][o->string_data[i].len] = 0;
                                        memcpy(str[i], o->string_data[i].data, o->string_data[i].len);
                                    }
                                }
                            }
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
                            n = MIN(t->ndata, (size_t)o->n_int64_data);
                            if ((n > 0) && t->datas && o->int64_data)
                                memcpy(t->datas, o->int64_data, sizeof(int64_t) * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
                            n = MIN(t->ndata, (size_t)o->n_double_data);
                            if ((n > 0) && t->datas && o->double_data)
                                memcpy(t->datas, o->double_data, sizeof(double) * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
                            // TODO
                            n = MIN(t->ndata, (size_t)o->n_uint64_data);
                            if ((n > 0) && t->datas && o->uint64_data)
                                memcpy(t->datas, o->uint64_data, sz * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
                            n = MIN(t->ndata, (size_t)(o->n_float_data / 2));
                            if ((n > 0) && t->datas && o->float_data)
                                memcpy(t->datas, o->float_data, sizeof(float) * 2 * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
                            n = MIN(t->ndata, (size_t)(o->n_double_data / 2));
                            if ((n > 0) && t->datas && o->double_data)
                                memcpy(t->datas, o->double_data, sizeof(double) * 2 * n);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

tensor_t *load_tensor_onnx(const char *path) {
    tensor_t *t = NULL;
    Onnx__TensorProto *pb;
    FILE *fp;
    void *buf;
    size_t l, len;
    int *dims = NULL;
    int ndim = 0;
    int i;

    fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        l = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        if (l > 0) {
            buf = sys_malloc(l);
            if (buf) {
                for (len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
                pb = onnx__tensor_proto__unpack(NULL, len, buf);
                sys_free(buf);
                if (pb) {
                    if (pb->n_dims > 0) {
                        dims = (int*)sys_malloc(sizeof(int) * pb->n_dims);
                        if (dims) {
                            for (i = 0; i < pb->n_dims; i++)
                                dims[i] = pb->dims[i];
                            ndim = pb->n_dims;
                        }
                    }
                    t = tensor_new(pb->name, (tensor_type_t)pb->data_type);
                    tensor_set_shape(t, ndim, dims);
                    if ((ndim > 0) && dims)
                        sys_free(dims);
                    tensor_copy_proto(t, pb);
                    onnx__tensor_proto__free_unpacked(pb, NULL);
                }
            }
        }
        fclose(fp);
    }
    return t;
}


graph_t *get_graph_onnx(context_t * ctx) {
    if(!ctx || !ctx->model) {
        return NULL;
    }
    EVO_UNUSED Onnx__GraphProto * graph = ((Onnx__ModelProto*)(ctx->model))->graph;
    EVO_UNUSED graph_t * g;
    EVO_UNUSED node_t * n;
    EVO_UNUSED tensor_t * t;
    EVO_UNUSED Onnx__TensorProto * o;
	EVO_UNUSED Onnx__ValueInfoProto * v;
	EVO_UNUSED char * p, * domain;
	EVO_UNUSED char * name;
	EVO_UNUSED int i, j, k, l;

    if(!graph)
        return NULL;

    g = (graph_t *)sys_malloc(sizeof(graph_t));
    if(!g)
        return NULL;
    memset(g, 0, sizeof(graph_t));

    g->nnode = graph->n_node;
    g->nodes = (node_t **)sys_malloc(sizeof(node_t *) * g->nnode);
    if(!g->nodes){
        sys_free(g);
        return NULL;
    }

    for(i = 0; i < graph->n_input; i++) {
        v = graph->input[i];
    }

    return g;
}

// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t *serializer_new() {
    serializer_t *s = (serializer_t *)sys_malloc(sizeof(serializer_t));
    // default load by onnx
    s->load = load_onnx;
    s->load_file = load_file_onnx;
    s->unload = unload_onnx;
    s->get_graph = NULL;
    return s;
}

void serializer_free(serializer_t *sez) {
    if (sez) {
        sez->load = NULL;
        sez->load_file = NULL;
        sez->get_graph = NULL;
        sez->unload = NULL;
        sys_free(sez);
        sez = NULL;
    }
}