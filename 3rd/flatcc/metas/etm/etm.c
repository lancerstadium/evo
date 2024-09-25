#include "etm_builder.h"
#include "etm_reader.h"
#include <stdio.h>


#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(etm, x)  // Specified in the schema.

int Model_create(flatcc_builder_t *B) {

    flatbuffers_uint16_vec_ref_t inTS = flatbuffers_uint16_vec_create(B, (const uint16_t[]){0}, 1);
    flatbuffers_uint16_vec_ref_t outTS = flatbuffers_uint16_vec_create(B, (const uint16_t[]){1}, 1);

    // Create Graph
    ns(Graph_ref_t) graph = etm_Graph_create(B, 0, inTS, outTS);


    // 6. Create Etm
    ns(Model_create_as_root(B, flatbuffers_string_create_str(B, "etm"), graph));

    return 0;
}


int Model_print(const void *buffer) {

    ns(Model_table_t) etm = ns(Model_as_root(buffer));
    if(etm == 0) return -1;

    printf("[Etm]: %s\n", ns(Model_name(etm)));

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