#include "etm_builder.h"
#include "etm_reader.h"
#include <stdio.h>


#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(etm, x)  // Specified in the schema.

int Model_create(flatcc_builder_t *B) {

    // create tensors
    etm_Tensor_ref_t ts1 = etm_Tensor_create(B,
        etm_TensorType_Uint8,
        flatbuffers_string_create(B, "ts1", 3), 
        flatbuffers_int32_vec_create(B, (const int32_t[]){1, 4}, 2),
        flatbuffers_uint8_vec_create(B, (const uint8_t[]){1, 2, 3, 4}, 4),
        4,
        0,
        0
    );
    etm_Tensor_ref_t ts2 = etm_Tensor_create(B, 
        etm_TensorType_Uint8,
        flatbuffers_string_create(B, "ts2", 3), 
        flatbuffers_int32_vec_create(B, (const int32_t[]){1, 4}, 2),
        flatbuffers_uint8_vec_create(B, (const uint8_t[]){4, 3, 2, 1}, 4),
        4,
        0,
        0
    );
    etm_Tensor_vec_ref_t ts = etm_Tensor_vec_create(B, (etm_Tensor_ref_t[]){ts1, ts2}, 2);

    // create nodes
    etm_Attr_ref_t attr1 = etm_Attr_create(B, 
        flatbuffers_string_create(B, "attr1", 5), 
        etm_AttrData_as_AttrDataInt(etm_AttrDataInt_create(B, 1))
    );
    etm_Attr_ref_t attr2 = etm_Attr_create(B, 
        flatbuffers_string_create(B, "attr2", 5), 
        etm_AttrData_as_AttrDataFloat(etm_AttrDataFloat_create(B, 3))
    );
    etm_Attr_vec_ref_t attrs = etm_Attr_vec_create(B, (etm_Attr_ref_t[]){attr1, attr2}, 2);
    etm_Node_ref_t nd1 = etm_Node_create(B,
        flatbuffers_string_create(B, "nd1", 3),
        flatbuffers_uint16_vec_create(B, (const uint16_t[]){0}, 1),
        flatbuffers_uint16_vec_create(B, (const uint16_t[]){1}, 1),
        etm_OpType_Add,
        attrs
    );
    etm_Node_vec_ref_t nd = etm_Node_vec_create(B, (etm_Node_ref_t[]){nd1}, 1);

    // create index
    flatbuffers_uint16_vec_ref_t inIdx = flatbuffers_uint16_vec_create(B, (const uint16_t[]){0}, 1);
    flatbuffers_uint16_vec_ref_t outIdx = flatbuffers_uint16_vec_create(B, (const uint16_t[]){1}, 1);

    // Create Graph
    etm_Graph_ref_t g1 = etm_Graph_create(B, ts, nd, 0, inIdx, outIdx);
    etm_Graph_vec_ref_t graph = etm_Graph_vec_create(B, (etm_Graph_ref_t[]){g1}, 1);

    // 6. Create Etm
    etm_Model_create_as_root(B, flatbuffers_string_create_str(B, "etm_model"), graph);

    return 0;
}


int Model_print(const void *buffer) {

    etm_Model_table_t etm = etm_Model_as_root(buffer);
    if(etm == 0) return -1;

    printf("[Etm]: %s\n", etm_Model_name(etm));

    etm_Graph_vec_t graphs = etm_Model_graphs(etm);
    for(int i = 0; i < etm_Graph_vec_len(graphs); i++) {
        etm_Graph_table_t g = etm_Graph_vec_at(graphs, i);
        etm_Tensor_vec_t tss = etm_Graph_tensors(g);
        etm_Node_vec_t nds = etm_Graph_nodes(g);
        // print tensor
        for(int j = 0; j < etm_Tensor_vec_len(tss); j++) {
            etm_Tensor_table_t ts = etm_Tensor_vec_at(tss, j);
            printf("%s <%d>\n", etm_Tensor_name(ts), etm_Tensor_type(ts));
        }
        // print node
        for(int j = 0; j < etm_Tensor_vec_len(nds); j++) {
            etm_Node_table_t nd = etm_Node_vec_at(nds, j);
            printf("%s <%d>\n", etm_Node_name(nd), etm_Node_optype(nd));
            etm_Attr_vec_t attrs = etm_Node_attrs(nd);
            for(int k = 0; k < etm_Attr_vec_len(attrs); k++) {
                etm_Attr_table_t attr = etm_Attr_vec_at(attrs, k);
                printf(" - %s [%d]\n", etm_Attr_k(attr), etm_Attr_v_type_get(attr));
            }
        }
    }

    return 0;
}

int main() {

    FILE* fp;
    char path[] = "demo.etm";

    // 1. Flat Buffer Init
    flatcc_builder_t builder;
    flatcc_builder_init(&builder);

    // 2. Alloc Buffer
    void *buf;
    size_t size;

    // 3. Create etm
    Model_create(&builder);

    // 4. Save etm
    buf = flatcc_builder_finalize_aligned_buffer(&builder, &size);
    fp = fopen(path, "wb");
    fwrite(buf, 1, size, fp);
    fclose(fp);
    flatcc_builder_aligned_free(buf);
    flatcc_builder_clear(&builder);

    // 5. Print etm
    fp = fopen(path, "rb");
    size = fread(buf, 1, size, fp);
    Model_print(buf);

    return 0;
}