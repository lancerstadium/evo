#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/math.h>
#include <evo/util/sys.h>

#include <math.h>
#include <string.h>

// ==================================================================================== //
//                                   tensor type API
// ==================================================================================== //

const char *tensor_type_tostring(tensor_type_t type) {
    static const char *typestr[17] = {
        "undefined",
        "float32",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "int32",
        "int64",
        "string",
        "bool",
        "float16",
        "float64",
        "uint32",
        "uint64",
        "complex64",
        "complex128",
        "bfloat16",
    };
    if ((type > 0) && (type < (sizeof(typestr) / sizeof((typestr)[0]))))
        return typestr[type];
    return typestr[0];
}

int tensor_type_sizeof(tensor_type_t type) {
    static const int typesz[17] = {
        0,
        sizeof(float),
        sizeof(uint8_t),
        sizeof(int8_t),
        sizeof(uint16_t),
        sizeof(int16_t),
        sizeof(int32_t),
        sizeof(int64_t),
        sizeof(char *),
        sizeof(uint8_t),
        sizeof(uint16_t),
        sizeof(double),
        sizeof(uint32_t),
        sizeof(uint64_t),
        sizeof(float) * 2,
        sizeof(double) * 2,
        sizeof(uint16_t),
    };
    if ((type > 0) && (type < (sizeof(typesz) / sizeof((typesz)[0]))))
        return typesz[type];
    return typesz[0];
}

// ==================================================================================== //
//                                   tensor API
// ==================================================================================== //

static inline const char *tensor_layout_tostring(uint8_t layout) {
    return (layout == 0) ? "NCHW" : "NHWC";
}

static inline void tensor_init(tensor_t *ts, int idx, int type) {
    // index & type & name
    ts->index = idx;
    ts->type = type;
    ts->name = NULL;
    ts->pnode = -1;
    // option
    ts->is_const = 0;       // ts is var
    ts->is_param = 0;       // ts is not param
    ts->is_ialloc = 1;      // ts is internal allocated
    ts->layout = 0;         // ts is layout NCHW
    // dim
    for (int i = 0; i < EVO_DIM_MAX; i++) {
        ts->dims[i] = 1;
        ts->strides[i] = 1;
    }
    ts->ndim = 0;
    // data
    ts->datas = NULL;
    // grad
    ts->grad = NULL;
}

tensor_t *tensor_new(const char *name, tensor_type_t type) {
    // ts init
    tensor_t *ts = (tensor_t *)sys_malloc(sizeof(tensor_t));
    if (!ts) return NULL;
    tensor_init(ts, -1, type);
    // name
    if (name) {
        const int str_len = align(strlen(name) + 1, EVO_ALIGN_SIZE);
        ts->name = (char *)sys_malloc(str_len);
        if (!ts->name) {
            sys_free(ts);
            return NULL;
        }
        memcpy(ts->name, name, str_len);
        ts->name[str_len - 1] = '\0';
    }
    return ts;
}

tensor_t *tensor_new_int64(const char *name, int* dims, int ndim, int64_t* is, size_t ni) {
    tensor_t *ts = tensor_new(name, TENSOR_TYPE_INT64);
    if (ts) {
        tensor_reshape(ts, ndim, dims);
        if (ni > 0 && is) {
            tensor_apply(ts, (void *)is, ni * sizeof(int64_t));
        } 
        return ts;
    }
    return NULL;
}

tensor_t *tensor_new_float32(const char *name, int* dims, int ndim, float* fs, size_t nf) {
    tensor_t *ts = tensor_new(name, TENSOR_TYPE_FLOAT32);
    if (ts) {
        tensor_reshape(ts, ndim, dims);
        if (nf > 0 && fs) {
            tensor_apply(ts, (void *)fs, nf * sizeof(float));
        } 
        return ts;
    }
    return NULL;
}

void tensor_fill_zero(tensor_t *ts) {
    if(!ts || ts->ndata == 0 || !ts->datas) return;
    switch(ts->type) {
        case TENSOR_TYPE_INT8: {
            int8_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int8_t)0;
            }
            break;
        }
        case TENSOR_TYPE_INT16: {
            int16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int16_t)0;
            }
            break;
        }
        case TENSOR_TYPE_INT32: {
            int32_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int32_t)0;
            }
            break;
        }
        case TENSOR_TYPE_INT64: {
            int64_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int64_t)0;
            }
            break;
        }
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = float32_to_bfloat16(0.0f);
            }
            break;            
        }
        case TENSOR_TYPE_FLOAT16: {
            uint16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = float32_to_float16(0.0f);
            }
            break;            
        }
        case TENSOR_TYPE_FLOAT32: {
            float* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = 0.0f;
            }
            break;
        }
        case TENSOR_TYPE_FLOAT64: {
            double* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = 0.0;
            }
            break;
        }
        default: {
            memset(ts->datas, 0, ts->ndata * tensor_type_sizeof(ts->type));
            break;
        }
    }
}

void tensor_fill_uniform(tensor_t *ts, float min_val, float max_val) {
    if(!ts || ts->ndata == 0 || !ts->datas) return;
    switch(ts->type) {
        case TENSOR_TYPE_INT8: {
            int8_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int8_t)(min_val + ((float) rand() / RAND_MAX) * (max_val - min_val));
            }
            break;
        }
        case TENSOR_TYPE_INT16: {
            int16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int16_t)(min_val + ((float) rand() / RAND_MAX) * (max_val - min_val));
            }
            break;
        }
        case TENSOR_TYPE_INT32: {
            int32_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int32_t)(min_val + ((float) rand() / RAND_MAX) * (max_val - min_val));
            }
            break;
        }
        case TENSOR_TYPE_INT64: {
            int64_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (int64_t)(min_val + ((float) rand() / RAND_MAX) * (max_val - min_val));
            }
            break;
        }
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = float32_to_bfloat16(min_val + ((float) rand() / RAND_MAX) * (max_val - min_val));
            }
            break;            
        }
        case TENSOR_TYPE_FLOAT16: {
            uint16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = float32_to_float16(min_val + ((float) rand() / RAND_MAX) * (max_val - min_val));
            }
            break;            
        }
        case TENSOR_TYPE_FLOAT32: {
            float* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = min_val + ((float) rand() / RAND_MAX) * (max_val - min_val);
            }
            break;
        }
        case TENSOR_TYPE_FLOAT64: {
            double* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                data[i] = (double)min_val + ((double) rand() / RAND_MAX) * ((double)max_val - (double)min_val);
            }
            break;
        }
        default: break;
    }
}


void tensor_fill_normal(tensor_t *ts, float mean, float stddev) {
    if(!ts || ts->ndata == 0 || !ts->datas) return;
    switch(ts->type) {
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                float u = ((float) rand() / RAND_MAX);
                float v = ((float) rand() / RAND_MAX);
                data[i] = float32_to_bfloat16(mean + stddev * sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v));
            }
            break;            
        }
        case TENSOR_TYPE_FLOAT16: {
            uint16_t* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                float u = ((float) rand() / RAND_MAX);
                float v = ((float) rand() / RAND_MAX);
                data[i] = float32_to_float16(mean + stddev * sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v));
            }
            break;            
        }
        case TENSOR_TYPE_FLOAT32: {
            float* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                float u = ((float) rand() / RAND_MAX);
                float v = ((float) rand() / RAND_MAX);
                data[i] = mean + stddev * sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v);
            }
            break;
        }
        case TENSOR_TYPE_FLOAT64: {
            double* data = ts->datas;
            for (int i = 0; i < ts->ndata; i++) {
                double u = ((double) rand() / RAND_MAX);
                double v = ((double) rand() / RAND_MAX);
                data[i] = mean + stddev * sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
            }
            break;
        }
        default: break;
    }
}


// Xavier Initialization: tanh & sigmoid
void tensor_fill_xavier(tensor_t *ts, int n_in, int n_out) {
    float stddev = sqrtf(2.0f / (n_in + n_out));  // Xavier's stddev
    tensor_fill_normal(ts, 0.0f, stddev);
}

// He Initialization: ReLU & Leaky ReLU
void tensor_fill_he(tensor_t *ts, int n_in) {
    float stddev = sqrtf(2.0f / n_in);  // He's stddev
    tensor_fill_normal(ts, 0.0f, stddev);
}

// LeCun Initialization: sigmoid & softmax
void tensor_fill_lecun(tensor_t *ts, int n_in) {
    float stddev = sqrtf(1.0f / n_in);  // LeCun's stddev
    tensor_fill_normal(ts, 0.0f, stddev);
}



void tensor_free(tensor_t *ts) {
    if (!ts) return;
    if (ts->name) sys_free(ts->name);
    if (ts->datas) sys_free(ts->datas);
    sys_free(ts);
    ts = NULL;
}

tensor_t *tensor_reinit(tensor_t *ts, tensor_type_t type, int ndim, int *dims) {
    char **str;
    int n;
    int sz, i;
    if (ts) {
        // release dim & data
        ts->ndim = 0;
        if ((ts->ndata > 0) && ts->datas) {
            if (ts->type == TENSOR_TYPE_STRING) {
                str = (char **)ts->datas;
                for (int idx = 0; idx < ts->ndata; idx++) {
                    if (str[idx]) {
                        free(str[idx]);
                        str[idx] = NULL;
                    }
                }
            }
            ts->datas = NULL;
            ts->ndata = 0;
        }
    }
    // reinit
    if (type != TENSOR_TYPE_UNDEFINED) {
        if ((ndim > 0) && dims) {
            // check size
            for (i = 0, n = 1; i < ndim; i++) {
                if (dims[i] <= 0)
                    return ts;
                n *= dims[i];
            }
            // init
            ts->type = type;
            ts->ndim = ndim;
            sz = tensor_type_sizeof(ts->type);
            if (n > 0 && sz > 0) {
                // ndim     = 3
                // dims     = [2,2,4]
                // strides  = [0,0,1]
                //          = [8,4,1]
                for (i = ts->ndim - 1; i >= 0; i--) {
                    ts->dims[i] = dims[i];
                    if (i == ts->ndim - 1) {
                        ts->strides[i] = 1;
                    } else {
                        ts->strides[i] = ts->dims[i + 1] * ts->strides[i + 1];
                    }
                }
                ts->datas = sys_malloc(n * sz);
                if (ts->datas) {
                    memset(ts->datas, 0, n * sz);
                    ts->ndata = n;
                }
            }
        } else {
            sz = tensor_type_sizeof(ts->type);
            if (sz > 0) {
                ts->datas = sys_malloc(sz);
                if (ts->datas) {
                    memset(ts->datas, 0, sz);
                    ts->ndata = 1;
                }
            }
        }
    }
    return ts;
}

tensor_t * tensor_new_one_hot(int ndim, int *dims, int label) {
    tensor_t* ts = tensor_new("one_hot", TENSOR_TYPE_FLOAT32);
    tensor_reshape(ts, ndim, dims);
    float* tsd = ts->datas;
    if(label >= 0) tsd[label - 1] = 1.0;
    return ts;
}

void permute_data(const void* src, void* dst, int* src_strides, int* dst_strides, int* sizes, int ndim, int element_size) {
    int indices[EVO_DIM_MAX] = {0};
    int total_elements = 1;
    for (int i = 0; i < ndim; ++i) {
        total_elements *= sizes[i];
    }
    
    for (int i = 0; i < total_elements; ++i) {
        int src_offset = 0;
        int dst_offset = 0;
        for (int j = 0; j < ndim; ++j) {
            src_offset += indices[j] * src_strides[j];
            dst_offset += indices[j] * dst_strides[j];
        }
        
        memcpy((char*)dst + dst_offset * element_size, (char*)src + src_offset * element_size, element_size);
        
        for (int j = ndim - 1; j >= 0; --j) {
            if (++indices[j] < sizes[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
}

tensor_t * tensor_permute(tensor_t *ts, int ndim, int* dim_idxs) {
    if(!ts || ndim < ts->ndim || !dim_idxs)
        return ts;
    int dims[ndim];
    int strides[ndim];
    for (int i = 0; i < ndim; i++) {
        dims[i] = ts->dims[dim_idxs[i]];
        strides[i] = ts->strides[dim_idxs[i]];
    }
    tensor_t * new_ts = tensor_new(ts->name, ts->type);
    tensor_reshape(new_ts, ndim, dims);
    // Compute the size of a single element
    int element_size = tensor_type_sizeof(ts->type);
    // Set new_ts->datas
    new_ts->ndata = ts->ndata;
    new_ts->datas = malloc(ts->ndata * element_size);
    if (!new_ts->datas) {
        free(new_ts);
        return NULL; // Memory allocation for data failed
    }
    // Compute the new strides for the permuted tensor
    int new_strides[EVO_DIM_MAX];
    new_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * dims[i + 1];
    }
    // Permute the data
    permute_data(ts->datas, new_ts->datas, strides, new_strides, dims, ndim, element_size);

    return new_ts;
}


tensor_t * tensor_argmax(tensor_t* ts, int axis, int keepdims, int select_last_index) {
    if(!ts) return ts;
    node_t* nd = node_temp("argmax", OP_TYPE_ARG_MAX);
    nd->nin = 1;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->out[0] = tensor_new("argmax_out", TENSOR_TYPE_FLOAT32);
    attribute_t* axis_attr = attribute_int("axis", axis);   // 0
    attribute_t* keepdims_attr = attribute_int("keepdims", keepdims);   // 1
    attribute_t* select_last_index_attr = attribute_int("select_last_index", axis); // 0
    vector_add(&nd->attr_vec, axis_attr);
    vector_add(&nd->attr_vec, keepdims_attr);
    vector_add(&nd->attr_vec, select_last_index_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

/** TODO: upsample */
tensor_t * tensor_resize(tensor_t* ts, int rz_w, int rz_h, char* mode) {
    if(!ts || ts->ndim != 4) return ts;
    node_t* nd = node_temp("resize", OP_TYPE_RESIZE);
    nd->nin = 4;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->in[1] = tensor_new("gather_roi", TENSOR_TYPE_FLOAT32);
    nd->in[2] = tensor_new("gather_scales", TENSOR_TYPE_FLOAT32);
    nd->out[3] = tensor_new("gather_sizes", TENSOR_TYPE_INT64);
    nd->out[0] = tensor_new("gather_out", TENSOR_TYPE_FLOAT32);
    tensor_reshape(nd->in[2], 2, (int[]){1, 4});
    float dims[4] = {1, 1, (float)rz_h / ts->dims[2], (float)rz_w / ts->dims[3]};
    tensor_apply(nd->in[2], dims, 4 * sizeof(float));
    attribute_t* mode_attr = attribute_string("mode", mode, strlen(mode));
    vector_add(&nd->attr_vec, mode_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {  
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_cast(tensor_t* ts, tensor_type_t type) {
    if(!ts) return ts;
    node_t* nd = node_temp("cast", OP_TYPE_CAST);
    nd->nin = 1;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->out[0] = tensor_new("cast_out", type);
    attribute_t* to_attr = attribute_int("to", type);
    vector_add(&nd->attr_vec, to_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_gather(tensor_t* ts, tensor_t* idx_ts, int axis) {
    if(!ts || !idx_ts) return ts;
    node_t* nd = node_temp("gather", OP_TYPE_GATHER);
    nd->nin = 2;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->in[1] = idx_ts;
    nd->out[0] = tensor_new("gather_out", TENSOR_TYPE_FLOAT32);
    attribute_t* axis_attr = attribute_int("axis", axis);
    vector_add(&nd->attr_vec, axis_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {  
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_scatternd(tensor_t* ts, tensor_t* idx_ts, tensor_t* upd_ts, char* reduction) {
    if(!ts || !idx_ts || !upd_ts) return ts;
    node_t* nd = node_temp("scatternd", OP_TYPE_SCATTER_ND);
    nd->nin = 3;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->in[1] = idx_ts;
    nd->in[2] = upd_ts;
    nd->out[0] = tensor_new("scatternd_out", TENSOR_TYPE_FLOAT32);
    attribute_t* reduction_attr = attribute_string("reduction", reduction, strlen(reduction));
    vector_add(&nd->attr_vec, reduction_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {  
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_expand(tensor_t* ts, int64_t* shps, size_t shps_size) {
    if(!ts || !shps) return ts;
    node_t* nd = node_temp("expand", OP_TYPE_EXPAND);
    nd->nin = 2;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->in[1] = tensor_new_int64("expand_shape", (int[]){1, shps_size}, 2, shps, shps_size);
    nd->out[0] = tensor_new("expand_out", TENSOR_TYPE_FLOAT32);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {  
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_pad(tensor_t* ts, int64_t* pads, size_t pads_size, char* mode) {
    if(!ts) return ts;
    node_t* nd = node_temp("pad", OP_TYPE_PAD);
    nd->nin = 2;
    if(strcmp(mode, "constant") == 0) nd->nin = 3;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->in[1] = tensor_new_int64("pad_pads", (int[]){1, 2 * ts->ndim}, 2, pads, pads_size);
    if(strcmp(mode, "constant") == 0) {
        nd->in[2] = tensor_new("pad_constant", ts->type);
        tensor_reshape(nd->in[2], 2, (int[]){1, 1});
    }
    nd->out[0] = tensor_new("pad_out", TENSOR_TYPE_FLOAT32);
    attribute_t* mode_attr = attribute_string("mode", mode, strlen(mode));
    vector_add(&nd->attr_vec, mode_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {  
        nd->op->init(nd);
        nd->op->reshape(nd);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_softmax(tensor_t* ts, int axis) {
    if(!ts) return ts;
    node_t* nd = node_temp("softmax", OP_TYPE_SOFTMAX);
    nd->nin = 1;
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    nd->out[0] = tensor_new("softmax_out", TENSOR_TYPE_FLOAT32);
    attribute_t* axis_attr = attribute_int("axis", axis);
    vector_add(&nd->attr_vec, axis_attr);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_squeeze(tensor_t* ts, int* axes, int axes_size) {
    if(!ts) return ts;
    node_t* nd = node_temp("squeeze", OP_TYPE_SQUEEZE);
    if(!axes || axes_size <= 0) {
        nd->nin = 1;
    } else {
        nd->nin = 2;
    }
    nd->nout= 1;
    nd->in = sys_malloc(nd->nin * sizeof(tensor_t*));
    nd->out = sys_malloc(nd->nout * sizeof(tensor_t*));
    nd->in[0] = ts;
    int64_t axes_is[axes_size];
    for(int i = 0; i < axes_size; i++) {
        axes_is[i] = axes[i];
    }
    nd->in[1] = tensor_new_int64("axes", (int[]){axes_size}, 1, axes_is, axes_size);
    nd->out[0] = tensor_new("squeeze_out", TENSOR_TYPE_FLOAT32);
    node_bind_op(nd);
    if(nd->op && nd->op->init) {
        nd->op->init(nd);
        nd->op->reshape(nd);
        nd->op->forward(nd);
        nd->op->exit(nd);
    }
    return nd->out[0];
}

tensor_t * tensor_nhwc2nchw(tensor_t *ts) {
    if(!ts || ts->layout == 0 || ts->ndim != 4) {
        return ts;
    }
    tensor_t* res = tensor_permute(ts, 4, (int[]){0, 3, 1, 2});
    res->layout = 0;
    return res;
}

tensor_t * tensor_nchw2nhwc(tensor_t *ts) {
    if(!ts || ts->layout == 1 || ts->ndim != 4) {
        return ts;
    }
    tensor_t* res = tensor_permute(ts, 4, (int[]){0, 2, 3, 1});
    res->layout = 1;
    return res;
}

bool tensor_equal(tensor_t *a, tensor_t *b) {
    size_t i;

    if (!a || !b)
        return 0;
    if (a->type != b->type)
        return 0;
    if (a->ndim != b->ndim)
        return 0;
    if (a->ndata != b->ndata)
        return 0;
    if (a->ndim > 0) {
        if (memcmp(a->dims, b->dims, sizeof(int) * a->ndim) != 0)
            return 0;
    }
    switch (a->type) {
        case TENSOR_TYPE_BOOL:
        case TENSOR_TYPE_INT8:
        case TENSOR_TYPE_INT16:
        case TENSOR_TYPE_INT32:
        case TENSOR_TYPE_INT64:
        case TENSOR_TYPE_UINT8:
        case TENSOR_TYPE_UINT16:
        case TENSOR_TYPE_UINT32:
        case TENSOR_TYPE_UINT64:
            if (memcmp(a->datas, b->datas, a->ndata * tensor_type_sizeof(a->type)) != 0)
                return 0;
            break;
        case TENSOR_TYPE_BFLOAT16: {
            uint16_t *p = (uint16_t *)a->datas;
            uint16_t *q = (uint16_t *)b->datas;
            for (i = 0; i < a->ndata; i++) {
                if (fabsf(bfloat16_to_float32(p[i]) - bfloat16_to_float32(q[i])) > 1e-3)
                    return 0;
            }
        } break;
        case TENSOR_TYPE_FLOAT16: {
            uint16_t *p = (uint16_t *)a->datas;
            uint16_t *q = (uint16_t *)b->datas;
            for (i = 0; i < a->ndata; i++) {
                if (fabsf(float16_to_float32(p[i]) - float16_to_float32(q[i])) > 1e-3)
                    return 0;
            }
        } break;
        case TENSOR_TYPE_FLOAT32: {
            float *p = (float *)a->datas;
            float *q = (float *)b->datas;
            for (i = 0; i < a->ndata; i++) {
                if (fabsf(p[i] - q[i]) > 1e-3)
                    return 0;
            }
        } break;
        case TENSOR_TYPE_FLOAT64: {
            double *p = (double *)a->datas;
            double *q = (double *)b->datas;
            for (i = 0; i < a->ndata; i++) {
                if (fabs(p[i] - q[i]) > 1e-3)
                    return 0;
            }
        } break;
        case TENSOR_TYPE_COMPLEX64: {
            float *p = (float *)a->datas;
            float *q = (float *)b->datas;
            for (i = 0; i < a->ndata * 2; i++) {
                if (fabsf(p[i] - q[i]) > 1e-3)
                    return 0;
            }
        } break;
        case TENSOR_TYPE_COMPLEX128: {
            double *p = (double *)a->datas;
            double *q = (double *)b->datas;
            for (i = 0; i < a->ndata * 2; i++) {
                if (fabs(p[i] - q[i]) > 1e-3)
                    return 0;
            }
        } break;
        case TENSOR_TYPE_STRING: {
            char **p = (char **)a->datas;
            char **q = (char **)b->datas;
            for (i = 0; i < a->ndata; i++) {
                if (p[i] && q[i] && (strcmp(p[i], q[i]) != 0))
                    return 0;
            }
        } break;
        default:
            break;
    }
    return 1;
}

int tensor_reshape(tensor_t *ts, int ndim, int *dims) {
    if (ndim > EVO_DIM_MAX) return -1;
    const int old_ndata = ts->ndata;
    int new_ndata = 1;
    for (int i = 0; i < ndim; i++) {
        ts->dims[i] = dims[i];
        new_ndata *= dims[i];
    }
    ts->ndim = ndim;
    ts->ndata = new_ndata;
    if (old_ndata != new_ndata) {
        tensor_reinit(ts, ts->type, ndim, dims);
    }
    return 0;
}

int tensor_reshape_ident(tensor_t *y, tensor_t *x, tensor_type_t type) {
    if ((y->ndim != x->ndim) || (memcmp(y->dims, x->dims, sizeof(int) * y->ndim) != 0) || (y->type != type))
        tensor_reinit(y, type, x->ndim, x->dims);
    return 1;
}

int tensor_reshape_multi_broadcast(tensor_t *y, tensor_t *a, tensor_t *b, tensor_type_t type) {
    int ndim = MAX(a->ndim, b->ndim);
    int dims[ndim];
    int i, j, k;
    if (ndim > 0) {
        for (i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--) {
            if (i < 0)
                dims[k] = b->dims[j--];
            else if (j < 0)
                dims[k] = a->dims[i--];
            else {
                if (a->dims[i] == b->dims[j])
                    dims[k] = a->dims[i];
                else if ((a->dims[i] == 1) || (b->dims[j] == 1))
                    dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
                else
                    return 0;
                i--;
                j--;
            }
        }
    }
    if ((y->type != type) || (y->ndim != ndim) || (memcmp(y->dims, dims, sizeof(int) * ndim != 0)))
        tensor_reinit(y, type, ndim, dims);
    return 1;
}

int tensor_broadcast_is_valid(tensor_t *x, int *dims, int ndim) {
    if(!x || x->ndim > ndim)
        return 0;
    int i;
    for (i = 1; i <= x->ndim; i++) {
        if ((x->dims[x->ndim - i] != 1) && (x->dims[x->ndim - i] != dims[ndim - i]))
            return 0;
    }
    return 1;
}

void *tensor_broadcast_map_address(tensor_t *x, tensor_t *y, int offset) {
    int xndim = x->ndim;
    int yndim = y->ndim;
    if ((x->ndim > 0) && (y->ndim > 0)) {
        int dndim = yndim - xndim;
        int ix[xndim];
        int iy[yndim];
        int i;
        tensor_offset2index(y, offset, iy);
        for (i = 0; i < xndim; i++)
            ix[i] = iy[dndim + i] % x->dims[i];
        return x->datas + tensor_index2offset(x, ix) * tensor_type_sizeof(x->type);
    }
    return x->datas;
}

int tensor_index2offset(tensor_t *ts, int *idxs) {
    int offset, i;
    for (i = 0, offset = 0; i < ts->ndim; i++)
        offset += idxs[i] * ts->strides[i];
    return offset;
}

void tensor_offset2index(tensor_t *ts, int offset, int *idxs) {
    int i;
    for (i = ts->ndim - 1; i >= 0; i--) {
        idxs[i] = offset % ts->dims[i];
        offset /= ts->dims[i];
    }
}

void tensor_apply(tensor_t *ts, void *buf, size_t len) {
    size_t l;
    int sz;
    if (ts) {
        if (ts->datas && buf && (len > 0)) {
            sz = tensor_type_sizeof(ts->type);
            if (sz > 0) {
                if (ts->type == TENSOR_TYPE_STRING) {
                    char **p = (char **)ts->datas;
                    char **q = (char **)buf;
                    for (int idx = 0; idx < ts->ndata; idx++) {
                        if (p[idx]) {
                            free(p[idx]);
                            p[idx] = NULL;
                        }
                    }
                    l = MIN(ts->ndata, (size_t)len);
                    for (int idx = 0; idx < l; idx++) {
                        p[idx] = sys_strdup(q[idx]);
                    }
                } else {
                    l = ts->ndata * sz;
                    if (l > 0) {
                        memcpy(ts->datas, buf, MIN(l, len));
                    }
                }
            }
        }
    }
}

void tensor_copy(tensor_t *a, tensor_t *b) {
    if(a && b) {
        tensor_apply(a, b->datas, b->ndata * tensor_type_sizeof(b->type));
    }
}

char *tensor_set_name_by_index(graph_t *g, int index) {
    char *name = (char *)sys_malloc(EVO_ALIGN_SIZE * 2);
    if (name) sprintf(name, "tensor_%d", index);
    return name;
}

int tensor_get_index_by_name(graph_t *g, const char *name) {
    const char *last_symbol_ptr = strrchr(name, '_');
    if (last_symbol_ptr) {
        const int index = atoi(last_symbol_ptr + 1);
        if (index >= 0 && index < g->ntensor) {
            const tensor_t *ts = g->tensors[index];
            if (ts && ts->name && strcmp(ts->name, name) == 0) {
                return index;
            }
        }
    }
    // search all names
    for (int i = 0; i < g->ntensor; i++) {
        if (g->tensors[i]->name && strcmp(g->tensors[i]->name, name) == 0) {
            return i;
        }
    }
    return -1;
}

char* tensor_dump_shape(tensor_t *ts) {
    if (!ts) return NULL;
    char* shape_buf = malloc(64 * sizeof(char));
    shape_buf[0] = '\0';
    if (ts->ndim > 0) {
        sprintf(shape_buf, "[");
        for (int i = 0; i < ts->ndim - 1; i++) {
            sprintf(shape_buf + strlen(shape_buf), "%d,", ts->dims[i]);
        }
        sprintf(shape_buf + strlen(shape_buf), "%d]", ts->dims[ts->ndim - 1]);
    } else {
        sprintf(shape_buf, "[]");
    }
    return shape_buf;
}

void tensor_dump(tensor_t *ts) {
    if (!ts) return;
    if (ts->name) {
        LOG_INFO("%s <%s>", strcmp(ts->name, "") == 0 ? "Tensor" : ts->name, tensor_type_tostring(ts->type), tensor_layout_tostring(ts->layout));
    }
    if (ts->ndim > 0) {
        char shape_buf[64];
        sprintf(shape_buf, " [");
        for (int i = 0; i < ts->ndim - 1; i++) {
            sprintf(shape_buf + strlen(shape_buf), "%d,", ts->dims[i]);
        }
        sprintf(shape_buf + strlen(shape_buf), "%d]", ts->dims[ts->ndim - 1]);
        LOG_INFO("%s", shape_buf);
    } else {
        LOG_INFO(" []");
    }
    LOG_INFO("\n");
}


void tensor_dump1(tensor_t *ts) {
    if (!ts) return;
    int *sizes, *levels;
    char *lbuf, *rbuf;
    char *lp, *rp;
    void *p;
    int i, j, k;
    if (ts->ndata > 1 && ts->datas) {
        for (i = 0; i < ts->ndim; i++) {
            if (ts->dims[i] <= 0)
                return;
        }
        sizes = malloc(sizeof(int) * ts->ndim);
        levels = malloc(sizeof(int) * ts->ndim);
        sizes[ts->ndim - 1] = ts->dims[ts->ndim - 1];
        levels[ts->ndim - 1] = 0;
        lbuf = malloc(sizeof(char) * (ts->ndim + 1));
        rbuf = malloc(sizeof(char) * (ts->ndim + 1));
        lp = lbuf;
        rp = rbuf;
        for (i = ts->ndim - 2; i >= 0; i--) {
            sizes[i] = ts->dims[i] * sizes[i + 1];
            levels[i] = 0;
        }
        for (size_t idx = 0; idx < ts->ndata; idx++) {
            for (j = 0; j < ts->ndim; j++) {
                if ((idx % sizes[j]) == 0)
                    levels[j]++;
                if (levels[j] == 1) {
                    *lp++ = '[';
                    levels[j]++;
                }
                if (levels[j] == 3) {
                    *rp++ = ']';
                    if ((j != 0) && (levels[j] > levels[j - 1])) {
                        *lp++ = '[';
                        levels[j] = 2;
                    } else {
                        levels[j] = 0;
                    }
                }
            }
            *lp = *rp = '\0';
            LOG_INFO("%s", rbuf);
            if (*rbuf != '\0') {
                LOG_INFO("\r\n");
                for (k = ts->ndim - strlen(rbuf); k > 0; k--)
                    LOG_INFO(" ");
            }
            LOG_INFO("%s", lbuf);
            if (*lbuf == '\0')
                LOG_INFO(" ");
            p = (void *)(ts->datas + tensor_type_sizeof(ts->type) * idx);
            switch (ts->type) {
                case TENSOR_TYPE_BOOL:
                    LOG_INFO("%s,", *((uint8_t *)p) ? "true" : "false");
                    break;
                case TENSOR_TYPE_INT8:
                    LOG_INFO("%d,", *((int8_t *)p));
                    break;
                case TENSOR_TYPE_INT16:
                    LOG_INFO("%d,", *((int16_t *)p));
                    break;
                case TENSOR_TYPE_INT32:
                    LOG_INFO("%d,", *((int32_t *)p));
                    break;
                case TENSOR_TYPE_INT64:
                    LOG_INFO("%ld,", *((int64_t *)p));
                    break;
                case TENSOR_TYPE_UINT8:
                    LOG_INFO("%u,", *((uint8_t *)p));
                    break;
                case TENSOR_TYPE_UINT16:
                    LOG_INFO("%u,", *((uint16_t *)p));
                    break;
                case TENSOR_TYPE_UINT32:
                    LOG_INFO("%u,", *((uint32_t *)p));
                    break;
                case TENSOR_TYPE_UINT64:
                    LOG_INFO("%lu,", *((uint64_t *)p));
                    break;
                case TENSOR_TYPE_BFLOAT16:
                    LOG_INFO("%g,", bfloat16_to_float32(*((uint16_t *)p)));
                    break;
                case TENSOR_TYPE_FLOAT16:
                    LOG_INFO("%g,", float16_to_float32(*((uint16_t *)p)));
                    break;
                case TENSOR_TYPE_FLOAT32:
                    LOG_INFO("%g,", *((float *)p));
                    break;
                case TENSOR_TYPE_FLOAT64:
                    LOG_INFO("%g,", *((double *)p));
                    break;
                case TENSOR_TYPE_COMPLEX64:
                    LOG_INFO("%g + %gi,", *((float *)p), *((float *)(p + sizeof(float))));
                    break;
                case TENSOR_TYPE_COMPLEX128:
                    LOG_INFO("%g + %gi,", *((double *)p), *((double *)(p + sizeof(double))));
                    break;
                case TENSOR_TYPE_STRING:
                    LOG_INFO("%s,", (char *)(((char **)p)[0]));
                    break;
                default:
                    LOG_INFO("?,");
                    break;
            }
            lp = lbuf;
            rp = rbuf;
        }
        for (j = 0; j < ts->ndim; j++)
            LOG_INFO("]");
        free(sizes);
        free(levels);
        free(lbuf);
        free(rbuf);
        LOG_INFO("\n");
    } else if (ts->ndata == 1 && ts->datas) {
        p = (void *)(ts->datas);
        switch (ts->type) {
            case TENSOR_TYPE_BOOL:
                LOG_INFO("%s", *((uint8_t *)p) ? "true" : "false");
                break;
            case TENSOR_TYPE_INT8:
                LOG_INFO("%d", *((int8_t *)p));
                break;
            case TENSOR_TYPE_INT16:
                LOG_INFO("%d", *((int16_t *)p));
                break;
            case TENSOR_TYPE_INT32:
                LOG_INFO("%d", *((int32_t *)p));
                break;
            case TENSOR_TYPE_INT64:
                LOG_INFO("%ld", *((int64_t *)p));
                break;
            case TENSOR_TYPE_UINT8:
                LOG_INFO("%u", *((uint8_t *)p));
                break;
            case TENSOR_TYPE_UINT16:
                LOG_INFO("%u", *((uint16_t *)p));
                break;
            case TENSOR_TYPE_UINT32:
                LOG_INFO("%u", *((uint32_t *)p));
                break;
            case TENSOR_TYPE_UINT64:
                LOG_INFO("%lu", *((uint64_t *)p));
                break;
            case TENSOR_TYPE_BFLOAT16:
                LOG_INFO("%g", bfloat16_to_float32(*((uint16_t *)p)));
                break;
            case TENSOR_TYPE_FLOAT16:
                LOG_INFO("%g", float16_to_float32(*((uint16_t *)p)));
                break;
            case TENSOR_TYPE_FLOAT32:
                LOG_INFO("%g", *((float *)p));
                break;
            case TENSOR_TYPE_FLOAT64:
                LOG_INFO("%g", *((double *)p));
                break;
            case TENSOR_TYPE_COMPLEX64:
                LOG_INFO("%g + %gi", *((float *)p), *((float *)(p + sizeof(float))));
                break;
            case TENSOR_TYPE_COMPLEX128:
                LOG_INFO("%g + %gi", *((double *)p), *((double *)(p + sizeof(double))));
                break;
            case TENSOR_TYPE_STRING:
                LOG_INFO("%s", (char *)(((char **)p)[0]));
                break;
            default:
                LOG_INFO("?");
                break;
        }
        LOG_INFO("\r\n");
    } else {
        LOG_INFO("[]\n");
    }
}

void tensor_dump2(tensor_t *ts) {
    if (!ts) return;
    int *sizes, *levels;
    char *lbuf, *rbuf;
    char *lp, *rp;
    void *p;
    int i, j, k;
    if (ts->name) {
        LOG_INFO("%s <%s/%s>", strcmp(ts->name, "") == 0 ? "Tensor" : ts->name, tensor_type_tostring(ts->type), tensor_layout_tostring(ts->layout));
    }
    if (ts->ndim > 0) {
        char shape_buf[64];
        sprintf(shape_buf, " [");
        for (int i = 0; i < ts->ndim - 1; i++) {
            sprintf(shape_buf + strlen(shape_buf), "%d,", ts->dims[i]);
        }
        sprintf(shape_buf + strlen(shape_buf), "%d]", ts->dims[ts->ndim - 1]);
        LOG_INFO("%s", shape_buf);
    } else {
        LOG_INFO(" []");
    }
    if (ts->ndata > 1 && ts->datas) {
        LOG_INFO(" = \n");
        for (i = 0; i < ts->ndim; i++) {
            if (ts->dims[i] <= 0)
                return;
        }
        sizes = malloc(sizeof(int) * ts->ndim);
        levels = malloc(sizeof(int) * ts->ndim);
        sizes[ts->ndim - 1] = ts->dims[ts->ndim - 1];
        levels[ts->ndim - 1] = 0;
        lbuf = malloc(sizeof(char) * (ts->ndim + 1));
        rbuf = malloc(sizeof(char) * (ts->ndim + 1));
        lp = lbuf;
        rp = rbuf;
        for (i = ts->ndim - 2; i >= 0; i--) {
            sizes[i] = ts->dims[i] * sizes[i + 1];
            levels[i] = 0;
        }
        for (size_t idx = 0; idx < ts->ndata; idx++) {
            for (j = 0; j < ts->ndim; j++) {
                if ((idx % sizes[j]) == 0)
                    levels[j]++;
                if (levels[j] == 1) {
                    *lp++ = '[';
                    levels[j]++;
                }
                if (levels[j] == 3) {
                    *rp++ = ']';
                    if ((j != 0) && (levels[j] > levels[j - 1])) {
                        *lp++ = '[';
                        levels[j] = 2;
                    } else {
                        levels[j] = 0;
                    }
                }
            }
            *lp = *rp = '\0';
            LOG_INFO("%s", rbuf);
            if (*rbuf != '\0') {
                LOG_INFO("\r\n");
                for (k = ts->ndim - strlen(rbuf); k > 0; k--)
                    LOG_INFO(" ");
            }
            LOG_INFO("%s", lbuf);
            if (*lbuf == '\0')
                LOG_INFO(" ");
            p = (void *)(ts->datas + tensor_type_sizeof(ts->type) * idx);
            switch (ts->type) {
                case TENSOR_TYPE_BOOL:
                    LOG_INFO("%s,", *((uint8_t *)p) ? "true" : "false");
                    break;
                case TENSOR_TYPE_INT8:
                    LOG_INFO("%d,", *((int8_t *)p));
                    break;
                case TENSOR_TYPE_INT16:
                    LOG_INFO("%d,", *((int16_t *)p));
                    break;
                case TENSOR_TYPE_INT32:
                    LOG_INFO("%d,", *((int32_t *)p));
                    break;
                case TENSOR_TYPE_INT64:
                    LOG_INFO("%ld,", *((int64_t *)p));
                    break;
                case TENSOR_TYPE_UINT8:
                    LOG_INFO("%u,", *((uint8_t *)p));
                    break;
                case TENSOR_TYPE_UINT16:
                    LOG_INFO("%u,", *((uint16_t *)p));
                    break;
                case TENSOR_TYPE_UINT32:
                    LOG_INFO("%u,", *((uint32_t *)p));
                    break;
                case TENSOR_TYPE_UINT64:
                    LOG_INFO("%lu,", *((uint64_t *)p));
                    break;
                case TENSOR_TYPE_BFLOAT16:
                    LOG_INFO("%g,", bfloat16_to_float32(*((uint16_t *)p)));
                    break;
                case TENSOR_TYPE_FLOAT16:
                    LOG_INFO("%g,", float16_to_float32(*((uint16_t *)p)));
                    break;
                case TENSOR_TYPE_FLOAT32:
                    LOG_INFO("%g,", *((float *)p));
                    break;
                case TENSOR_TYPE_FLOAT64:
                    LOG_INFO("%g,", *((double *)p));
                    break;
                case TENSOR_TYPE_COMPLEX64:
                    LOG_INFO("%g + %gi,", *((float *)p), *((float *)(p + sizeof(float))));
                    break;
                case TENSOR_TYPE_COMPLEX128:
                    LOG_INFO("%g + %gi,", *((double *)p), *((double *)(p + sizeof(double))));
                    break;
                case TENSOR_TYPE_STRING:
                    LOG_INFO("%s,", (char *)(((char **)p)[0]));
                    break;
                default:
                    LOG_INFO("?,");
                    break;
            }
            lp = lbuf;
            rp = rbuf;
        }
        for (j = 0; j < ts->ndim; j++)
            LOG_INFO("]");
        free(sizes);
        free(levels);
        free(lbuf);
        free(rbuf);
        LOG_INFO("\n");
    } else if (ts->ndata == 1 && ts->datas) {
        LOG_INFO(" = ");
        p = (void *)(ts->datas);
        switch (ts->type) {
            case TENSOR_TYPE_BOOL:
                LOG_INFO("%s", *((uint8_t *)p) ? "true" : "false");
                break;
            case TENSOR_TYPE_INT8:
                LOG_INFO("%d", *((int8_t *)p));
                break;
            case TENSOR_TYPE_INT16:
                LOG_INFO("%d", *((int16_t *)p));
                break;
            case TENSOR_TYPE_INT32:
                LOG_INFO("%d", *((int32_t *)p));
                break;
            case TENSOR_TYPE_INT64:
                LOG_INFO("%ld", *((int64_t *)p));
                break;
            case TENSOR_TYPE_UINT8:
                LOG_INFO("%u", *((uint8_t *)p));
                break;
            case TENSOR_TYPE_UINT16:
                LOG_INFO("%u", *((uint16_t *)p));
                break;
            case TENSOR_TYPE_UINT32:
                LOG_INFO("%u", *((uint32_t *)p));
                break;
            case TENSOR_TYPE_UINT64:
                LOG_INFO("%lu", *((uint64_t *)p));
                break;
            case TENSOR_TYPE_BFLOAT16:
                LOG_INFO("%g", bfloat16_to_float32(*((uint16_t *)p)));
                break;
            case TENSOR_TYPE_FLOAT16:
                LOG_INFO("%g", float16_to_float32(*((uint16_t *)p)));
                break;
            case TENSOR_TYPE_FLOAT32:
                LOG_INFO("%g", *((float *)p));
                break;
            case TENSOR_TYPE_FLOAT64:
                LOG_INFO("%g", *((double *)p));
                break;
            case TENSOR_TYPE_COMPLEX64:
                LOG_INFO("%g + %gi", *((float *)p), *((float *)(p + sizeof(float))));
                break;
            case TENSOR_TYPE_COMPLEX128:
                LOG_INFO("%g + %gi", *((double *)p), *((double *)(p + sizeof(double))));
                break;
            case TENSOR_TYPE_STRING:
                LOG_INFO("%s", (char *)(((char **)p)[0]));
                break;
            default:
                LOG_INFO("?");
                break;
        }
        LOG_INFO("\r\n");
    } else {
        LOG_INFO(" = []\n");
    }
    if (ts->pnode >= 0) {
        LOG_INFO(" from node: %d", ts->pnode);
    }
    LOG_INFO("\n");
}


char* tensor_to_string(tensor_t *ts) {
    if (!ts) return NULL;
    char* result = (char*)malloc(1024 * sizeof(char)); // Initial buffer size
    if (!result) return NULL;
    int buf_size = 1024;
    int offset = 0;

    int *sizes, *levels;
    char *lbuf, *rbuf;
    char *lp, *rp;
    void *p;
    int i, j, k;
    
    offset += snprintf(result + offset, buf_size - offset, "%s <%s/%s>",
                       strcmp(ts->name, "") == 0 ? "Tensor" : ts->name,
                       tensor_type_tostring(ts->type),
                       tensor_layout_tostring(ts->layout));

    if (ts->ndim > 0) {
        char shape_buf[64];
        sprintf(shape_buf, " [");
        for (int i = 0; i < ts->ndim - 1; i++) {
            sprintf(shape_buf + strlen(shape_buf), "%d,", ts->dims[i]);
        }
        sprintf(shape_buf + strlen(shape_buf), "%d]", ts->dims[ts->ndim - 1]);
        offset += snprintf(result + offset, buf_size - offset, "%s", shape_buf);
    } else {
        offset += snprintf(result + offset, buf_size - offset, " []");
    }

    if (ts->ndata > 1 && ts->datas) {
        offset += snprintf(result + offset, buf_size - offset, " = \n");
        for (i = 0; i < ts->ndim; i++) {
            if (ts->dims[i] <= 0) {
                free(result);
                return NULL;
            }
        }
        sizes = malloc(sizeof(int) * ts->ndim);
        levels = malloc(sizeof(int) * ts->ndim);
        sizes[ts->ndim - 1] = ts->dims[ts->ndim - 1];
        levels[ts->ndim - 1] = 0;
        lbuf = malloc(sizeof(char) * (ts->ndim + 1));
        rbuf = malloc(sizeof(char) * (ts->ndim + 1));
        lp = lbuf;
        rp = rbuf;
        for (i = ts->ndim - 2; i >= 0; i--) {
            sizes[i] = ts->dims[i] * sizes[i + 1];
            levels[i] = 0;
        }
        for (size_t idx = 0; idx < ts->ndata; idx++) {
            for (j = 0; j < ts->ndim; j++) {
                if ((idx % sizes[j]) == 0)
                    levels[j]++;
                if (levels[j] == 1) {
                    *lp++ = '[';
                    levels[j]++;
                }
                if (levels[j] == 3) {
                    *rp++ = ']';
                    if ((j != 0) && (levels[j] > levels[j - 1])) {
                        *lp++ = '[';
                        levels[j] = 2;
                    } else {
                        levels[j] = 0;
                    }
                }
            }
            *lp = *rp = '\0';
            offset += snprintf(result + offset, buf_size - offset, "%s", rbuf);
            if (*rbuf != '\0') {
                offset += snprintf(result + offset, buf_size - offset, "\n");
                for (k = ts->ndim - strlen(rbuf); k > 0; k--)
                    offset += snprintf(result + offset, buf_size - offset, " ");
            }
            offset += snprintf(result + offset, buf_size - offset, "%s", lbuf);
            if (*lbuf == '\0')
                offset += snprintf(result + offset, buf_size - offset, " ");
            p = (void *)(ts->datas + tensor_type_sizeof(ts->type) * idx);
            switch (ts->type) {
                case TENSOR_TYPE_BOOL:
                    offset += snprintf(result + offset, buf_size - offset, "%s,", *((uint8_t *)p) ? "true" : "false");
                    break;
                case TENSOR_TYPE_INT8:
                    offset += snprintf(result + offset, buf_size - offset, "%d,", *((int8_t *)p));
                    break;
                case TENSOR_TYPE_INT16:
                    offset += snprintf(result + offset, buf_size - offset, "%d,", *((int16_t *)p));
                    break;
                case TENSOR_TYPE_INT32:
                    offset += snprintf(result + offset, buf_size - offset, "%d,", *((int32_t *)p));
                    break;
                case TENSOR_TYPE_INT64:
                    offset += snprintf(result + offset, buf_size - offset, "%ld,", *((int64_t *)p));
                    break;
                case TENSOR_TYPE_UINT8:
                    offset += snprintf(result + offset, buf_size - offset, "%u,", *((uint8_t *)p));
                    break;
                case TENSOR_TYPE_UINT16:
                    offset += snprintf(result + offset, buf_size - offset, "%u,", *((uint16_t *)p));
                    break;
                case TENSOR_TYPE_UINT32:
                    offset += snprintf(result + offset, buf_size - offset, "%u,", *((uint32_t *)p));
                    break;
                case TENSOR_TYPE_UINT64:
                    offset += snprintf(result + offset, buf_size - offset, "%lu,", *((uint64_t *)p));
                    break;
                case TENSOR_TYPE_BFLOAT16:
                    offset += snprintf(result + offset, buf_size - offset, "%g,", bfloat16_to_float32(*((uint16_t *)p)));
                    break;
                case TENSOR_TYPE_FLOAT16:
                    offset += snprintf(result + offset, buf_size - offset, "%g,", float16_to_float32(*((uint16_t *)p)));
                    break;
                case TENSOR_TYPE_FLOAT32:
                    offset += snprintf(result + offset, buf_size - offset, "%g,", *((float *)p));
                    break;
                case TENSOR_TYPE_FLOAT64:
                    offset += snprintf(result + offset, buf_size - offset, "%g,", *((double *)p));
                    break;
                case TENSOR_TYPE_COMPLEX64:
                    offset += snprintf(result + offset, buf_size - offset, "%g + %gi,", *((float *)p), *((float *)(p + sizeof(float))));
                    break;
                case TENSOR_TYPE_COMPLEX128:
                    offset += snprintf(result + offset, buf_size - offset, "%g + %gi,", *((double *)p), *((double *)(p + sizeof(double))));
                    break;
                case TENSOR_TYPE_STRING:
                    offset += snprintf(result + offset, buf_size - offset, "%s,", (char *)(((char **)p)[0]));
                    break;
                default:
                    offset += snprintf(result + offset, buf_size - offset, "?,");
                    break;
            }
            lp = lbuf;
            rp = rbuf;
        }
        for (j = 0; j < ts->ndim; j++)
            offset += snprintf(result + offset, buf_size - offset, "]");
        free(sizes);
        free(levels);
        free(lbuf);
        free(rbuf);
        offset += snprintf(result + offset, buf_size - offset, "\n");
    } else if (ts->ndata == 1 && ts->datas) {
        offset += snprintf(result + offset, buf_size - offset, " = ");
        p = (void *)(ts->datas);
        switch (ts->type) {
            case TENSOR_TYPE_BOOL:
                offset += snprintf(result + offset, buf_size - offset, "%s", *((uint8_t *)p) ? "true" : "false");
                break;
            case TENSOR_TYPE_INT8:
                offset += snprintf(result + offset, buf_size - offset, "%d", *((int8_t *)p));
                break;
            case TENSOR_TYPE_INT16:
                offset += snprintf(result + offset, buf_size - offset, "%d", *((int16_t *)p));
                break;
            case TENSOR_TYPE_INT32:
                offset += snprintf(result + offset, buf_size - offset, "%d", *((int32_t *)p));
                break;
            case TENSOR_TYPE_INT64:
                offset += snprintf(result + offset, buf_size - offset, "%ld", *((int64_t *)p));
                break;
            case TENSOR_TYPE_UINT8:
                offset += snprintf(result + offset, buf_size - offset, "%u", *((uint8_t *)p));
                break;
            case TENSOR_TYPE_UINT16:
                offset += snprintf(result + offset, buf_size - offset, "%u", *((uint16_t *)p));
                break;
            case TENSOR_TYPE_UINT32:
                offset += snprintf(result + offset, buf_size - offset, "%u", *((uint32_t *)p));
                break;
            case TENSOR_TYPE_UINT64:
                offset += snprintf(result + offset, buf_size - offset, "%lu", *((uint64_t *)p));
                break;
            case TENSOR_TYPE_BFLOAT16:
                offset += snprintf(result + offset, buf_size - offset, "%g", bfloat16_to_float32(*((uint16_t *)p)));
                break;
            case TENSOR_TYPE_FLOAT16:
                offset += snprintf(result + offset, buf_size - offset, "%g", float16_to_float32(*((uint16_t *)p)));
                break;
            case TENSOR_TYPE_FLOAT32:
                offset += snprintf(result + offset, buf_size - offset, "%g", *((float *)p));
                break;
            case TENSOR_TYPE_FLOAT64:
                offset += snprintf(result + offset, buf_size - offset, "%g", *((double *)p));
                break;
            case TENSOR_TYPE_COMPLEX64:
                offset += snprintf(result + offset, buf_size - offset, "%g + %gi", *((float *)p), *((float *)(p + sizeof(float))));
                break;
            case TENSOR_TYPE_COMPLEX128:
                offset += snprintf(result + offset, buf_size - offset, "%g + %gi", *((double *)p), *((double *)(p + sizeof(double))));
                break;
            case TENSOR_TYPE_STRING:
                offset += snprintf(result + offset, buf_size - offset, "%s,", (char *)(((char **)p)[0]));
                break;
            default:
                offset += snprintf(result + offset, buf_size - offset, "?,");
                break;
        }
    } else {
        offset += snprintf(result + offset, buf_size - offset, " = []");
    }
    return result;
}