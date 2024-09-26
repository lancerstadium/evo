#include <evo/mdl/tflite/tflite.h>
#include <evo/mdl/tflite/tflite_builder.h>
#include <evo/mdl/tflite/tflite_reader.h>
#include <evo/util/sys.h>
#include <evo/util/log.h>
#include <evo/util/math.h>
#include <stdio.h>

/**
 * tflite mobilenet: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
 * 
 */

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(tflite, x)  // Specified in the schema.

EVO_UNUSED static flatcc_builder_t builder;

model_t *load_tflite(struct serializer *s, const void *buf, size_t len) {
    model_t *mdl = NULL;
    if (!buf || len <= 0)
        return NULL;
    mdl = model_new(NULL);
    mdl->sez = s;
    mdl->cmodel = ns(Model_as_root(buf));
    if (!mdl->cmodel) {
        if (mdl)
            sys_free(mdl);
        return NULL;
    }
    mdl->model_size = len;
    mdl->name = sys_strdup(ns(Model_description(mdl->cmodel)));
    mdl->tensor_map = hashmap_create();
    if (!mdl->tensor_map) {
        if (mdl)
            sys_free(mdl);
        return NULL;
    }
    // graph
    load_graph_tflite(mdl);
    return mdl;
}

model_t *load_model_tflite(struct serializer *sez, const char *path) {
    model_t *mdl = NULL;
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
                mdl = load_tflite(sez, buf, len);
                sys_free(buf);
            }
        }
        fclose(fp);
    } else {
        LOG_ERR("No such file: %s\n", path);
    }
    return mdl;
}

void unload_tflite(model_t *mdl) {
    if (mdl && mdl->cmodel) {
        mdl->model_size = 0;
    }
}

EVO_UNUSED tensor_t *load_tensor_tflite(const char *path);

static inline tensor_type_t tensor_type_map(ns(TensorType_enum_t) type) {
    switch (type) {
        case ns(TensorType_FLOAT16): return TENSOR_TYPE_FLOAT16;
        case ns(TensorType_FLOAT32): return TENSOR_TYPE_FLOAT32;
        case ns(TensorType_FLOAT64): return TENSOR_TYPE_FLOAT64;
        case ns(TensorType_UINT8): return TENSOR_TYPE_UINT8;
        case ns(TensorType_UINT16): return TENSOR_TYPE_UINT16;
        case ns(TensorType_UINT32): return TENSOR_TYPE_UINT32;
        case ns(TensorType_UINT64): return TENSOR_TYPE_UINT64;
        case ns(TensorType_INT8): return TENSOR_TYPE_INT8;
        case ns(TensorType_INT32): return TENSOR_TYPE_INT32;
        case ns(TensorType_INT64): return TENSOR_TYPE_INT64;
        case ns(TensorType_STRING): return TENSOR_TYPE_STRING;
        case ns(TensorType_COMPLEX64): return TENSOR_TYPE_COMPLEX64;
        case ns(TensorType_COMPLEX128): return TENSOR_TYPE_COMPLEX128;
        default: return TENSOR_TYPE_UNDEFINED;
    }
}

static op_type_t op_map_tflite(ns(OperatorCode_table_t) opcode) {
    switch(ns(OperatorCode_builtin_code(opcode))) {
        case ns(BuiltinOperator_ABS):               return OP_TYPE_ABS;
        case ns(BuiltinOperator_ADD):               return OP_TYPE_ADD;
        case ns(BuiltinOperator_ADD_N):             return OP_TYPE_ADD;
        case ns(BuiltinOperator_ARG_MAX):           return OP_TYPE_ARG_MAX;
        case ns(BuiltinOperator_ARG_MIN):           return OP_TYPE_ARG_MIN;
        case ns(BuiltinOperator_AVERAGE_POOL_2D):   return OP_TYPE_MAX_POOL;
        case ns(BuiltinOperator_CONV_2D):           return OP_TYPE_CONV;
        case ns(BuiltinOperator_DEPTHWISE_CONV_2D): return OP_TYPE_CONV;
        case ns(BuiltinOperator_MEAN):              return OP_TYPE_MEAN;
        case ns(BuiltinOperator_FULLY_CONNECTED):   return OP_TYPE_GEMM;
        case ns(BuiltinOperator_SHAPE):             return OP_TYPE_SHAPE;
        case ns(BuiltinOperator_SOFTMAX):           return OP_TYPE_SOFTMAX;
        case ns(BuiltinOperator_RELU):              return OP_TYPE_RELU;
        default: return OP_TYPE_NOP;
    }
}

static tensor_t * tensor_from_proto(ns(Tensor_table_t) tensor) {
    tensor_t * ts = tensor_new(sys_strdup(ns(Tensor_name(tensor))), tensor_type_map(ns(Tensor_type(tensor))));
    int dims[EVO_DIM_MAX];
    int i;
    flatbuffers_int32_vec_t tdims  = ns(Tensor_shape(tensor));
    int ndim = (int)flatbuffers_int32_vec_len(tdims);
    for(i = 0; i < MIN(ndim, EVO_DIM_MAX); i++) {
        dims[i] = flatbuffers_int32_vec_at(tdims, i);
    }
    tensor_reshape(ts, ndim, dims);
    ts->layout = 1; // default: NHWC
    return ts;
}

graph_t *load_graph_tflite(model_t *mdl) {
    if (!mdl || !mdl->cmodel) {
        return NULL;
    }
    graph_t *g;
    ns(SubGraph_vec_t) subgraphs = ns(Model_subgraphs(mdl->cmodel));
    if (!subgraphs)
        return NULL;
    g = graph_new(mdl);
    if (!g)
        return NULL;
    // Print cur Operator codes
    ns(OperatorCode_vec_t) opcodes = ns(Model_operator_codes(mdl->cmodel));
    for(size_t i = 0; i < ns(OperatorCode_vec_len(opcodes)); i++) {
        ns(OperatorCode_table_t) opcode = ns(OperatorCode_vec_at(opcodes, i));
        LOG_INFO("opcode: %d\n", ns(OperatorCode_builtin_code(opcode)));
    }
    // New Subgraph: subgraph <== tflite subgraph
    for(size_t i = 0; i < ns(SubGraph_vec_len(subgraphs)); i++) {
        ns(SubGraph_table_t) subgraph = ns(SubGraph_vec_at(subgraphs, i));
        graph_t *sg = graph_sub_new(g);
        sg->name = sys_strdup(ns(SubGraph_name(subgraph)));
        // Add Tensors: NHWC
        ns(Tensor_vec_t) tensors = ns(SubGraph_tensors(subgraph));
        tensor_vec_t sgts_vec = vector_create();
        for(size_t j = 0; j < ns(Tensor_vec_len(tensors)); j++) {
            ns(Tensor_table_t) tensor = ns(Tensor_vec_at(tensors, j));
            tensor_t * t = tensor_from_proto(tensor);
            vector_add(&sgts_vec, t);
            hashmap_set(mdl->tensor_map, hashmap_str_lit(t->name), (uintptr_t)t);
        }
        // Add Nodes: from Operator
        ns(Operator_vec_t) operators = ns(SubGraph_operators(subgraph));
        node_vec_t sgnd_vec = vector_create();
        for(size_t j = 0; j < ns(Operator_vec_len(operators)); j++) {
            ns(Operator_table_t) operator = ns(Operator_vec_at(operators, j));
            op_type_t optype = op_map_tflite(ns(OperatorCode_vec_at(opcodes, ns(Operator_opcode_index(operator)))));
            char nd_name[32];
            sprintf(nd_name, "%s_%s%d", sg->name, op_name(optype), (int)vector_size(sgnd_vec));
            node_t * nd = node_new(sg, nd_name, optype);
            flatbuffers_int32_vec_t inputs = ns(Operator_inputs(operator));
            flatbuffers_int32_vec_t outputs = ns(Operator_outputs(operator));
            size_t ninput = flatbuffers_int32_vec_len(inputs);
            size_t noutput = flatbuffers_int32_vec_len(outputs);
            nd->in = (tensor_t **)sys_malloc(ninput * sizeof(tensor_t*));
            nd->out = (tensor_t **)sys_malloc(noutput * sizeof(tensor_t*));
            int32_t last_idx = 0;
            if(nd->in) {
                nd->nin = ninput;
                for(size_t k = 0; k < nd->nin; k++) {
                    nd->in[k] = sgts_vec[flatbuffers_int32_vec_at(inputs, k)];
                }
            }
            if(nd->out) {
                nd->nout = noutput;
                for(size_t k = 0; k < nd->nout; k++) {
                    nd->out[k] = sgts_vec[flatbuffers_int32_vec_at(outputs, k)];
                    if(k == 0) last_idx = k;
                }
            }
            vector_add(&sgnd_vec, nd);
            if(nd->op->type == OP_TYPE_CONV) {
                ns(Conv2DOptions_table_t) opt = ns(Operator_builtin_options(operator));
                // int sw = ns(Conv2DOptions_stride_w)(opt);
                // int sh = ns(Conv2DOptions_stride_h)(opt);
                op_type_t act_optype = OP_TYPE_NOP;
                switch(ns(Conv2DOptions_fused_activation_function(opt))){
                    case ns(ActivationFunctionType_RELU): act_optype = OP_TYPE_RELU; break;
                    case ns(ActivationFunctionType_NONE):
                    default: break;
                }
                if(act_optype != OP_TYPE_NOP) {
                    sprintf(nd_name, "%s_%s%d", sg->name, op_name(act_optype), (int)vector_size(sgnd_vec));
                    node_t* nda = node_new(sg, nd_name, act_optype);
                    nda->in = (tensor_t **)sys_malloc(1 * sizeof(tensor_t*));
                    nda->out = (tensor_t **)sys_malloc(1 * sizeof(tensor_t*));
                    nda->in = (tensor_t **)sys_malloc(1 * sizeof(tensor_t*));
                    nda->out = (tensor_t **)sys_malloc(1 * sizeof(tensor_t*));
                    if(nda->out) {
                        nda->nout = 1;
                        nda->out[0] = sgts_vec[flatbuffers_int32_vec_at(inputs, last_idx)];
                    }
                    if(nda->in) {
                        sprintf(nd_name, "%s_%s%d_out0", sg->name, op_name(optype), (int)vector_size(sgnd_vec) - 1);
                        nda->nin = 1;
                        nda->in[0] = tensor_new(nd_name, nda->out[0]->type);
                        tensor_reshape(nda->in[0], nda->out[0]->ndim, nda->out[0]->dims);
                        vector_add(&sgts_vec, nda->out[0]);
                        hashmap_set(mdl->tensor_map, hashmap_str_lit(nda->in[0]->name), (uintptr_t)nda->in[0]);
                        nd->out[0] = nda->in[0];
                    }
                    vector_add(&sgnd_vec, nda);
                }
            }
        }
        // node
        sg->nnode = vector_size(sgnd_vec);
        sg->nodes = (node_t **)sys_malloc(sizeof(node_t *) * sg->nnode);
        if(!sg->nodes) {
            sys_free(sg);
            return NULL;
        }
        for(int i = 0; i < sg->nnode; i++) {
            sg->nodes[i] = sgnd_vec[i];
        }
        // tensor
        sg->ntensor = vector_size(sgts_vec);
        sg->tensors = (tensor_t **)sys_malloc(sizeof(tensor_t *) * sg->ntensor);
        if(!sg->tensors) {
            sys_free(sg);
            return NULL;
        }
        vector_free(sgnd_vec);
        vector_free(sgts_vec);
        // Add subgraph
        vector_add(&(g->sub_vec), sg);
    }
    return g;
}

EVO_UNUSED void save_tflite(model_t *mdl, const char* path) {

}


#undef ns