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

EVO_UNUSED model_t *load_etm(struct serializer *s, const void *buf, size_t len) {}
EVO_UNUSED model_t *load_model_etm(struct serializer *sez, const char *path) {}
EVO_UNUSED void unload_etm(model_t *mdl) {}
EVO_UNUSED tensor_t *load_tensor_etm(const char *path) {}
EVO_UNUSED graph_t *load_graph_etm(model_t *mdl) {}

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
                attribute_t* attr = nd->attr_vec[i];
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