#include <evo/mdl/etm/etm.h>
#include <evo/mdl/etm/etm.flat.h>
#include <evo/util/sys.h>
#include <evo/util/log.h>
#include <evo/util/math.h>
#include <stdio.h>

static inline tensor_type_t tensor_type_map(etm_TensorType_enum_t type) {
    switch (type) {
        case etm_TensorType_Float16     : return TENSOR_TYPE_FLOAT16;
        case etm_TensorType_Float32     : return TENSOR_TYPE_FLOAT32;
        case etm_TensorType_Float64     : return TENSOR_TYPE_FLOAT64;
        case etm_TensorType_Uint8       : return TENSOR_TYPE_UINT8;
        case etm_TensorType_Uint16      : return TENSOR_TYPE_UINT16;
        case etm_TensorType_Uint32      : return TENSOR_TYPE_UINT32;
        case etm_TensorType_Uint64      : return TENSOR_TYPE_UINT64;
        case etm_TensorType_Int8        : return TENSOR_TYPE_INT8;
        case etm_TensorType_Int32       : return TENSOR_TYPE_INT32;
        case etm_TensorType_Int64       : return TENSOR_TYPE_INT64;
        case etm_TensorType_String      : return TENSOR_TYPE_STRING;
        case etm_TensorType_Complex64   : return TENSOR_TYPE_COMPLEX64;
        case etm_TensorType_Complex128  : return TENSOR_TYPE_COMPLEX128;
        default: return TENSOR_TYPE_UNDEFINED;
    }
}

static inline etm_TensorType_enum_t tensor_type_map2(tensor_type_t type) {
    switch (type) {
        case TENSOR_TYPE_FLOAT16        : return etm_TensorType_Float16;
        case TENSOR_TYPE_FLOAT32        : return etm_TensorType_Float32;
        case TENSOR_TYPE_FLOAT64        : return etm_TensorType_Float64;
        case TENSOR_TYPE_UINT8          : return etm_TensorType_Uint8;
        case TENSOR_TYPE_UINT16         : return etm_TensorType_Uint16;
        case TENSOR_TYPE_UINT32         : return etm_TensorType_Uint32;
        case TENSOR_TYPE_UINT64         : return etm_TensorType_Uint64;
        case TENSOR_TYPE_INT8           : return etm_TensorType_Int8;
        case TENSOR_TYPE_INT32          : return etm_TensorType_Int32;
        case TENSOR_TYPE_INT64          : return etm_TensorType_Int64;
        case TENSOR_TYPE_STRING         : return etm_TensorType_String;
        case TENSOR_TYPE_COMPLEX64      : return etm_TensorType_Complex64;
        case TENSOR_TYPE_COMPLEX128     : return etm_TensorType_Complex128;
        default: return etm_TensorType_Undefined;
    }
}

static tensor_t* tensor_from_proto(etm_Tensor_table_t tsp) {
    tensor_t * ts = tensor_new(etm_Tensor_name(tsp), tensor_type_map(etm_Tensor_type(tsp)));
    int dims[EVO_DIM_MAX];
    int i;
    flatbuffers_int32_vec_t tdims  = etm_Tensor_dims(tsp);
    int ndim = (int)flatbuffers_int32_vec_len(tdims);
    for(i = 0; i < MIN(ndim, EVO_DIM_MAX); i++) {
        dims[i] = flatbuffers_int32_vec_at(tdims, i);
    }
    tensor_reshape(ts, ndim, dims);
    ts->layout = 0; // default: NCHW
    return ts;
}

static attribute_t* attr_from_proto(etm_Attr_table_t attrp) {
    switch(etm_Attr_v_type(attrp)) {
        case etm_AttrData_AttrDataFloat: {
            etm_AttrDataFloat_table_t dt = etm_Attr_v(attrp);
            return attribute_float((char*)etm_Attr_k(attrp), etm_AttrDataFloat_f(dt));
        }
        case etm_AttrData_AttrDataInt: {
            etm_AttrDataInt_table_t dt = etm_Attr_v(attrp);
            return attribute_int((char*)etm_Attr_k(attrp), etm_AttrDataInt_i(dt));
        }
        case etm_AttrData_AttrDataString: {
            etm_AttrDataString_table_t dt = etm_Attr_v(attrp);
            flatbuffers_string_t ss = etm_AttrDataString_s(dt);
            return attribute_string((char*)etm_Attr_k(attrp), (char*)ss, flatbuffers_string_len(ss));
        }
        case etm_AttrData_AttrDataFloats: {
            etm_AttrDataFloats_table_t dt = etm_Attr_v(attrp);
            flatbuffers_float_vec_t fs = etm_AttrDataFloats_fs(dt);
            size_t nf = flatbuffers_float_vec_len(fs);
            float fss[nf];
            for(int i = 0; i < nf; i++) {
                fss[i] = flatbuffers_float_vec_at(fs, i);
            }
            return attribute_floats((char*)etm_Attr_k(attrp), fss, nf);
        }
        case etm_AttrData_AttrDataInts: {
            etm_AttrDataInts_table_t dt = etm_Attr_v(attrp);
            flatbuffers_int64_vec_t is = etm_AttrDataInts_is(dt);
            size_t ni = flatbuffers_int64_vec_len(is);
            int64_t iss[ni];
            for(int i = 0; i < ni; i++) {
                iss[i] = flatbuffers_int64_vec_at(is, i);
            }
            return attribute_ints((char*)etm_Attr_k(attrp), iss, ni);
        }
        default: return NULL;
    }
}


model_t *load_etm(struct serializer *s, const void *buf, size_t len) {
    model_t *mdl = NULL;
    if (!buf || len <= 0)
        return NULL;
    mdl = model_new(NULL);
    mdl->sez = s;
    mdl->cmodel = etm_Model_as_root(buf);
    if (!mdl->cmodel) {
        if (mdl)
            sys_free(mdl);
        return NULL;
    }
    mdl->model_size = len;
    mdl->name = sys_strdup(etm_Model_name(mdl->cmodel));
    mdl->tensor_map = hashmap_create();
    if (!mdl->tensor_map) {
        if (mdl)
            sys_free(mdl);
        return NULL;
    }
    load_graph_etm(mdl);
    return mdl;
}

model_t *load_model_etm(struct serializer *sez, const char *path) {
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
                mdl = load_etm(sez, buf, len);
                sys_free(buf);
            }
        }
        fclose(fp);
    } else {
        LOG_ERR("No such file: %s\n", path);
    }
    return mdl;
}

void unload_etm(model_t *mdl) {
    if (mdl) {
        if(mdl->cmodel) {
            mdl->model_size = 0;
        }
        if(mdl->graph) {
            graph_free(mdl->graph);
        }
        mdl->cmodel = NULL;
        free(mdl);
        mdl = NULL;
    }
}
EVO_UNUSED tensor_t *load_tensor_etm(const char *path) {
    return NULL;
}

graph_t *load_graph_etm(model_t *mdl) {
    if (!mdl || !mdl->cmodel) {
        return NULL;
    }
    char name_buf[54];
    etm_Graph_vec_t s_sgs = etm_Model_graphs(mdl->cmodel);
    graph_t *g = graph_new(mdl);
    if(!s_sgs || !g) return NULL;
    for(size_t i = 0; i < etm_Graph_vec_len(s_sgs); i++) {
        etm_Graph_table_t s_sg = etm_Graph_vec_at(s_sgs, i);
        graph_t *sg = graph_sub_new(g);
        sprintf(name_buf, "%s%lu", etm_Model_name(mdl->cmodel), i);
        // Add Tensors: NCHW
        etm_Tensor_vec_t s_tss = etm_Graph_tensors(s_sg);
        sg->ntensor = etm_Tensor_vec_len(s_tss);
        sg->tensors = (tensor_t **)sys_malloc(sizeof(tensor_t *) * sg->ntensor);
        if(!sg->tensors) {
            sys_free(sg);
            return NULL;
        }
        for(size_t j = 0; j < sg->ntensor; j++) {
            etm_Tensor_table_t s_ts = etm_Tensor_vec_at(s_tss, j);
            sg->tensors[j] = tensor_from_proto(s_ts);
            hashmap_set(mdl->tensor_map, hashmap_str_lit(sg->tensors[j]->name), (uintptr_t)sg->tensors[j]);
        }
        // Add Nodes
        etm_Node_vec_t s_nds = etm_Graph_nodes(s_sg);
        sg->nnode = etm_Node_vec_len(s_nds);
        sg->nodes = (node_t **)sys_malloc(sizeof(node_t *) * sg->nnode);
        if(!sg->nodes) {
            sys_free(sg);
            return NULL;
        }
        for(size_t j = 0; j < sg->nnode; j++) {
            etm_Node_table_t s_nd = etm_Node_vec_at(s_nds, j);
            node_t * nd = node_new(sg, etm_Node_name(s_nd), (op_type_t)etm_Node_optype(s_nd));
            // index
            flatbuffers_uint16_vec_t s_in = etm_Node_in(s_nd);
            flatbuffers_uint16_vec_t s_out = etm_Node_out(s_nd);
            size_t ninput = flatbuffers_uint16_vec_len(s_in);
            size_t noutput = flatbuffers_uint16_vec_len(s_out);
            nd->in = (tensor_t **)sys_malloc(ninput * sizeof(tensor_t*));
            nd->out = (tensor_t **)sys_malloc(noutput * sizeof(tensor_t*));
            if(nd->in) {
                nd->nin = ninput;
                for(size_t k = 0; k < nd->nin; k++) {
                    nd->in[k] = sg->tensors[flatbuffers_uint16_vec_at(s_in, k)];
                }
            }
            if(nd->out) {
                nd->nout = noutput;
                for(size_t k = 0; k < nd->nout; k++) {
                    nd->out[k] = sg->tensors[flatbuffers_uint16_vec_at(s_out, k)];
                }
            }
            // Add Attrs
            etm_Attr_vec_t s_attrs = etm_Node_attrs(s_nd);
            size_t nattr = etm_Attr_vec_len(s_attrs);
            for(size_t k = 0; k < nattr; k++) {
                etm_Attr_table_t s_attr = etm_Attr_vec_at(s_attrs, k);
                attribute_t * attr = attr_from_proto(s_attr);
                vector_add(&nd->attr_vec, attr);
            }
            sg->nodes[j] = nd;
        }
    }
    return g;
}

void save_etm(model_t *mdl, const char* path) {
    if(!mdl || !mdl->graph || vector_size(mdl->graph->sub_vec) <= 0) return;
    if(!path) {
        char name_buf[54];
        sprintf(name_buf, "%s.etm", mdl->name ? mdl->name : "model");
        path = sys_strdup(name_buf);
    }
    FILE *fp;
    void *buf;
    size_t size;
    tensor_t* ts;
    node_t* nd;
    graph_t* g;
    flatcc_builder_t B;

    LOG_INFO("model name save: %s\n", mdl->name);

    // Create
    graph_t* pg = mdl->graph;
    etm_Graph_ref_t s_g[vector_size(pg->sub_vec)];
    flatcc_builder_init(&B);

    // Create Graphs
    for(int a = 0; a < vector_size(pg->sub_vec); a++) {
        g = pg->sub_vec[a];
        // Create Indexs
        uint16_t s_in[vector_size(g->inx)];
        uint16_t s_out[vector_size(g->outx)];
        flatbuffers_uint16_vec_ref_t s_ins = flatbuffers_uint16_vec_create(&B, s_in, vector_size(g->inx));
        flatbuffers_uint16_vec_ref_t s_outs = flatbuffers_uint16_vec_create(&B, s_out, vector_size(g->outx));
        // Create Tensors
        etm_Tensor_ref_t s_ts[g->ntensor];
        for(int i = 0; i < g->ntensor; i++) {
            ts = g->tensors[i];
            s_ts[i] = etm_Tensor_create(&B, 
                tensor_type_map2(ts->type),
                flatbuffers_string_create_str(&B, ts->name), 
                flatbuffers_int32_vec_create(&B, ts->dims, ts->ndim),
                flatbuffers_uint8_vec_create(&B, ts->datas, ts->ndata * tensor_type_sizeof(ts->type)),
                ts->ndata,
                ts->is_param,
                ts->layout
            );
        }
        etm_Tensor_vec_ref_t s_tss = etm_Tensor_vec_create(&B, s_ts, g->ntensor);
        // Create Nodes
        etm_Node_ref_t s_nd[g->nnode];
        for(int i = 0; i < g->nnode; i++) {
            nd = g->nodes[i];
            uint16_t in[nd->nin];
            uint16_t out[nd->nout];
            etm_Attr_ref_t s_attr[vector_size(nd->attr_vec)];
            for(int j = 0; j < nd->nin; j++) {
                in[j] = nd->in[j]->index;
            }
            for(int j = 0; j < nd->nout; j++) {
                out[j] = nd->out[j]->index;
            }
            for(int j = 0; j < vector_size(nd->attr_vec); j++) {
                attribute_t* attr = nd->attr_vec[j];
                switch(attr->type) {
                    case ATTRIBUTE_TYPE_FLOAT: {
                        s_attr[j] = etm_Attr_create(&B, 
                            flatbuffers_string_create_str(&B, attr->name), 
                            etm_AttrData_as_AttrDataFloat(etm_AttrDataFloat_create(&B, attr->f))
                        );
                        break;
                    }
                    case ATTRIBUTE_TYPE_INT: {
                        s_attr[j] = etm_Attr_create(&B, 
                            flatbuffers_string_create_str(&B, attr->name), 
                            etm_AttrData_as_AttrDataInt(etm_AttrDataInt_create(&B, attr->i))
                        );
                        break;
                    };
                    case ATTRIBUTE_TYPE_STRING: {
                        s_attr[j] = etm_Attr_create(&B, 
                            flatbuffers_string_create_str(&B, attr->name),
                            etm_AttrData_as_AttrDataString(etm_AttrDataString_create(&B, flatbuffers_string_create(&B, attr->ss, attr->ns)))
                        );
                        break;
                    };
                    case ATTRIBUTE_TYPE_FLOATS: {
                        s_attr[j] = etm_Attr_create(&B, 
                            flatbuffers_string_create_str(&B, attr->name), 
                            etm_AttrData_as_AttrDataFloats(etm_AttrDataFloats_create(&B, flatbuffers_float_vec_create(&B, (const float*)attr->fs, attr->nf)))
                        );
                        break;
                    }
                    case ATTRIBUTE_TYPE_INTS: {
                        s_attr[j] = etm_Attr_create(&B, 
                            flatbuffers_string_create_str(&B, attr->name), 
                            etm_AttrData_as_AttrDataInts(etm_AttrDataInts_create(&B, flatbuffers_int64_vec_create(&B, (const int64_t*)attr->is, attr->ni)))
                        );
                        break;
                    };
                    default: break;
                }
            }
            etm_Attr_vec_ref_t s_attrs = etm_Attr_vec_create(&B, s_attr, vector_size(nd->attr_vec));
            s_nd[i] = etm_Node_create(&B,
                flatbuffers_string_create_str(&B, nd->name),
                flatbuffers_uint16_vec_create(&B, (const uint16_t*)in, nd->nin),
                flatbuffers_uint16_vec_create(&B, (const uint16_t*)out, nd->nout),
                (etm_OpType_enum_t)nd->op->type,
                s_attrs
            );
        }
        etm_Node_vec_ref_t s_nds = etm_Node_vec_create(&B, s_nd, g->nnode);
        // Create Graph
        s_g[a] = etm_Graph_create(&B, s_tss, s_nds, g->data_layout, s_ins, s_outs);
    }
    etm_Model_create_as_root(&B, flatbuffers_string_create_str(&B, mdl->name), etm_Graph_vec_create(&B, s_g, vector_size(pg->sub_vec)));

    buf = flatcc_builder_finalize_aligned_buffer(&B, &size);
    fp = fopen(path, "wb");
    fwrite(buf, 1, size, fp);
    fclose(fp);
    flatcc_builder_aligned_free(buf);
    flatcc_builder_clear(&B);
}