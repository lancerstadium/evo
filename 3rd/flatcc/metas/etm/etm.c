#include "etm_builder.h"
#include "etm_reader.h"
#include <stdio.h>


#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(etm, x)  // Specified in the schema.

int Model_create(flatcc_builder_t *B) {

    // 3. Create Op
    flatbuffers_string_ref_t name1 = flatbuffers_string_create_str(B, "ConvOp");
    flatbuffers_string_ref_t name2 = flatbuffers_string_create_str(B, "PoolOp");
    
    int32_t inputIndexes1[] = {0};
    int32_t outputIndexs1[] = {1};
    int32_t inputIndexes2[] = {1};
    int32_t outputIndexs2[] = {2};

    ns(Op_ref_t) op1 = ns(Op_create(B, ns(OpType_Conv), name1, flatbuffers_int32_vec_create(B, inputIndexes1, 1), flatbuffers_int32_vec_create(B, outputIndexs1, 1)));
    ns(Op_ref_t) op2 = ns(Op_create(B, ns(OpType_Pool), name2, flatbuffers_int32_vec_create(B, inputIndexes2, 1), flatbuffers_int32_vec_create(B, outputIndexs2, 1)));

    // 4. Create Op list
    ns(Op_vec_start(B));
    ns(Op_vec_push(B, op1));
    ns(Op_vec_push(B, op2));
    ns(Op_vec_ref_t) oplist = ns(Op_vec_end(B));


    // 5. Create tensorName

    flatbuffers_string_ref_t tensorName1 = flatbuffers_string_create_str(B, "input");
    flatbuffers_string_ref_t tensorName2 = flatbuffers_string_create_str(B, "Output");

    flatbuffers_string_vec_start(B);
    flatbuffers_string_vec_push(B, tensorName1);
    flatbuffers_string_vec_push(B, tensorName2);
    flatbuffers_string_vec_ref_t tensorNameList = flatbuffers_string_vec_end(B);


    // 6. Create Etm
    ns(Model_create_as_root(B, flatbuffers_string_create_str(B, "etm"), oplist, tensorNameList));

    return 0;
}


int Model_print(const void *buffer) {

    ns(Model_table_t) etm = ns(Model_as_root(buffer));
    if(etm == 0) return -1;

    printf("[Etm]: %s\n", ns(Model_name(etm)));
    // Output Op list
    ns(Op_vec_t) oplists = ns(Model_oplists(etm));
    size_t oplist_len = ns(Op_vec_len(oplists));
    for(size_t i = 0; i < oplist_len; i++) {
        ns(Op_table_t) op = ns(Op_vec_at(oplists, i));
        printf("   [%2ld] Op Name: %s, Type: %d\n", i, ns(Op_name(op)), ns(Op_type(op)));
    }
    // Output Tensor list
    printf("   Tensors: [");
    flatbuffers_string_vec_t tensor_names = ns(Model_tensorName(etm));
    size_t tensor_name_len = flatbuffers_string_vec_len(tensor_names);
    for(size_t i = 0; i < tensor_name_len; i++) {
        printf("%s,", flatbuffers_string_vec_at(tensor_names, i));
    }
    printf("]\n");
    return 0;
}

int main() {

    // 1. Flat Buffer Init
    flatcc_builder_t builder;
    flatcc_builder_init(&builder);

    // 2. Alloc Buffer
    void *buf;
    size_t size;

    // 3. Create etm
    Model_create(&builder);

    // 4. Print etm
    buf = flatcc_builder_finalize_aligned_buffer(&builder, &size);
    Model_print(buf);
    flatcc_builder_aligned_free(buf);

    // -1. Flat Buffer Release
    flatcc_builder_clear(&builder);
    printf("Hello: etm!\n");
    return 0;
}