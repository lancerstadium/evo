
#include "evo.h"
#include "sys.h"
#include "log.h"
#include "math.h"
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

static inline const char* tensor_layout_tostring(uint8_t layout) {
    return (layout == 0) ? "NCHW" : "NHWC";
}

static inline void tensor_init(tensor_t *ts, int idx, int type) {
    // index & type & name
    ts->index = idx;
    ts->type = type;
    ts->name = NULL;
    ts->nelem = 0;
    ts->szelem = tensor_type_sizeof(type);
    ts->pnode = -1;
    // option
    ts->is_reshaped = 0;    // ts is not reshaped
    ts->is_constant = 0;    // ts is var
    ts->is_iallocated = 1;  // ts is internal allocated
    ts->layout = 0;         // ts is layout NCHW
    // dim
    for(int i = 0; i < EVO_DIM_MAX; i++) {
        ts->dims[i] = 1;
        ts->strides[i] = 1;
    }
    ts->ndim = 0;
    // data
    ts->datas = NULL;
}

tensor_t *tensor_new(const char *name, tensor_type_t type) {
    // ts init
    tensor_t *ts = (tensor_t *)sys_malloc(sizeof(tensor_t));
    if (!ts) return NULL;
    tensor_init(ts, -1, type);
    // name
    if(name) {
        const int str_len = align(strlen(name) + 1, EVO_ALIGN_SIZE);
        ts->name = (char *)sys_malloc(str_len);
        if(!ts->name) {
            sys_free(ts);
            return NULL;
        }
        memcpy(ts->name, name, str_len);
        ts->name[str_len - 1] = '\0';
    }
    return ts;
}


void tensor_free(tensor_t* ts) {
    if(!ts) return;
    if(ts->name) sys_free(ts->name);
    if(ts->datas) sys_free(ts->datas);
    sys_free(ts);
    ts = NULL;
}

tensor_t * tensor_reinit(tensor_t *ts, tensor_type_t type, int ndim, int *dims) {
    char ** str;
    int n;
    int sz, i;
    if(ts) {
        // release dim & data
        ts->ndim = 0;
        if((ts->ndata > 0) && ts->datas) {
            if(ts->type == TENSOR_TYPE_STRING) {
                str = (char**)ts->datas;
                for(int idx = 0; idx < ts->ndata; idx++) {
                    if(str[idx]) {
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
    if(type != TENSOR_TYPE_UNDEFINED) {
        if((ndim > 0) && dims) {
            // check size
            for(i = 0, n = 1; i < ndim; i++) {
                if(dims[i] <= 0)
                    return ts;
                n *= dims[i];
            }
            // init
            ts->type = type;
            ts->ndim = ndim;
            sz = tensor_type_sizeof(ts->type);
            if(n > 0 && sz > 0) {
                // ndim     = 3
                // dims     = [2,2,4]
                // strides  = [0,0,1]
                //          = [8,4,1]
                for(i = ts->ndim - 1; i >= 0; i--) {
                    ts->dims[i] = dims[i];
                    if(i == ts->ndim - 1) {
                        ts->strides[i] = 1;
                    } else {
                        ts->strides[i] = ts->dims[i+1] * ts->strides[i+1];
                    }
                }
                ts->datas = sys_malloc(n * sz);
                if(ts->datas) {
                    memset(ts->datas, 0, n * sz);
                    ts->ndata = n;
                }
            }
        } else {
            sz = tensor_type_sizeof(ts->type);
            if(sz > 0) {
                ts->datas = sys_malloc(sz);
                if(ts->datas) {
                    memset(ts->datas, 0, sz);
                    ts->ndata = 1;
                }
            }
        }
    }
    return ts;
}


int tensor_set_shape(tensor_t *ts, int ndim, int *dims) {
    if(ndim > EVO_DIM_MAX) return -1;
    const int old_nelem = ts->nelem;
    int new_nelem = 1;
    for(int i = 0; i < ndim; i++) {
        ts->dims[i] = dims[i];
        new_nelem *= dims[i];
    }
    ts->ndim = ndim;
    ts->nelem = new_nelem;
    if(old_nelem != new_nelem) {
        /// TODO: reshape
    }
    return 0;
}

char* tensor_set_name_by_index(graph_t *g, int index) {
    char* name = (char*)sys_malloc(EVO_ALIGN_SIZE * 2);
    if(name) sprintf(name, "tensor_%d", index);
    return name;
}

int tensor_get_index_by_name(graph_t *g, const char *name) {
    const char* last_symbol_ptr = strrchr(name, '_');
    if(last_symbol_ptr) {
        const int index = atoi(last_symbol_ptr + 1);
        if(index >= 0 && index < g->ntensor) {
            const tensor_t* ts = g->tensors[index];
            if(ts && ts->name && strcmp(ts->name, name) == 0) {
                return index;
            }
        }
    }
    // search all names
    for(int i = 0; i < g->ntensor; i++) {
        if(g->tensors[i]->name && strcmp(g->tensors[i]->name, name) == 0) {
            return i;
        }
    }
    return -1;
}


void tensor_dump(tensor_t *ts) {
    if(!ts) return;
    if(ts->name) {
        LOG_INFO("%s type: %s/%s", ts->name, tensor_type_tostring(ts->type), tensor_layout_tostring(ts->layout));
    }
    if(ts->ndim > 0) {
        char shape_buf[64];
        sprintf(shape_buf, " shape: [");
        for(int i = 0; i < ts->ndim - 1; i++) {
            sprintf(shape_buf + strlen(shape_buf), "%d,", ts->dims[i]);
        }
        sprintf(shape_buf + strlen(shape_buf), "%d]", ts->dims[ts->ndim - 1]);
        LOG_INFO("%s", shape_buf);
    } else {
        LOG_INFO(" shape: []");
    }
    if(ts->pnode >= 0) {
        LOG_INFO(" from node: %d", ts->pnode);
    }
    LOG_INFO("\n");
}