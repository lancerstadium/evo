#include "sob.h"
#include <evo.h>

#define MD(s)       "node/"s"/model.onnx"
#define TI(s, i)    "node/"s"/test_data_set_0/input_"#i".pb"
#define TO(s, i)    "node/"s"/test_data_set_0/output_"#i".pb"

#define TESTD_1I(s, i, o)                         \
    context_t* ctx = sez->load_model(sez, MD(s)); \
    tensor_t* i = context_get_tensor(ctx, #i);    \
    tensor_t* o = context_get_tensor(ctx, #o);    \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));    \
    tensor_t* t1 = sez->load_tensor(TO(s, 0));    \
    tensor_copy(i, t0);                           \
    graph_prerun(ctx->graph);                     \
    graph_run(ctx->graph);                        \
    tensor_dump2(o);                              \
    tensor_dump2(t1);                             \
    ctx->sez->unload(ctx);

#define TEST_1I(s, i, o)                            \
    context_t* ctx = sez->load_model(sez, MD(s));   \
    tensor_t* i = context_get_tensor(ctx, #i);      \
    tensor_t* o = context_get_tensor(ctx, #o);      \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));      \
    tensor_t* t1 = sez->load_tensor(TO(s, 0));      \
    tensor_copy(i, t0);                             \
    graph_prerun(ctx->graph);                       \
    graph_run(ctx->graph);                          \
    UnitTest_ast(tensor_equal(o, t1), s " failed"); \
    ctx->sez->unload(ctx);

#define TESTD_2I(s, i0, i1, o)                    \
    context_t* ctx = sez->load_model(sez, MD(s)); \
    tensor_t* i0 = context_get_tensor(ctx, #i0);  \
    tensor_t* i1 = context_get_tensor(ctx, #i1);  \
    tensor_t* o = context_get_tensor(ctx, #o);    \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));    \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));    \
    tensor_t* t2 = sez->load_tensor(TO(s, 0));    \
    tensor_copy(i0, t0);                          \
    tensor_copy(i1, t1);                          \
    graph_prerun(ctx->graph);                     \
    graph_run(ctx->graph);                        \
    tensor_dump2(o);                              \
    tensor_dump2(t2);                             \
    ctx->sez->unload(ctx);

#define TEST_2I(s, i0, i1, o)                       \
    context_t* ctx = sez->load_model(sez, MD(s));   \
    tensor_t* i0 = context_get_tensor(ctx, #i0);    \
    tensor_t* i1 = context_get_tensor(ctx, #i1);    \
    tensor_t* o = context_get_tensor(ctx, #o);      \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));      \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));      \
    tensor_t* t2 = sez->load_tensor(TO(s, 0));      \
    tensor_copy(i0, t0);                            \
    tensor_copy(i1, t1);                            \
    graph_prerun(ctx->graph);                       \
    graph_run(ctx->graph);                          \
    UnitTest_ast(tensor_equal(o, t2), s " failed"); \
    ctx->sez->unload(ctx);


serializer_t * sez;

// ---------------------- Init  ----------------------

UnitTest_fn_def(test_node_init) {
    device_reg("cpu");
    sez = serializer_new("onnx");
    return NULL;
}

// ---------------------- Abs    ----------------------

UnitTest_fn_def(test_abs) {
    TEST_1I("test_abs", x, y);
    return NULL;
}

// ---------------------- Add    ----------------------

UnitTest_fn_def(test_add) {
    TEST_2I("test_add", x, y, sum);
    return NULL;
}

// ---------------------- Conv  ----------------------

UnitTest_fn_def(test_conv_ap) {
    TEST_2I("test_conv_with_autopad_same", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv_sap) {
    TEST_2I("test_conv_with_strides_and_asymmetric_padding", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv_snp) {
    TEST_2I("test_conv_with_strides_no_padding", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv_sp) {
    TEST_2I("test_conv_with_strides_padding", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv) {
    UnitTest_add(test_conv_ap);
    UnitTest_add(test_conv_sap);
    UnitTest_add(test_conv_snp);
    UnitTest_add(test_conv_sp);
    return NULL;
}

// ---------------------- MatMul  ----------------------

UnitTest_fn_def(test_matmul_2d) {
    TEST_2I("test_matmul_2d", a, b, c);
    return NULL;
}
UnitTest_fn_def(test_matmul_3d) {
    TEST_2I("test_matmul_3d", a, b, c);
    return NULL;
}
UnitTest_fn_def(test_matmul_4d) {
    TEST_2I("test_matmul_4d", a, b, c);
    return NULL;
}
UnitTest_fn_def(test_matmul) {
    UnitTest_add(test_matmul_2d);
    UnitTest_add(test_matmul_3d);
    UnitTest_add(test_matmul_4d);
    return NULL;
}

// ---------------------- MaxPool ----------------------

UnitTest_fn_def(test_maxpool_1d) {
    TEST_1I("test_maxpool_1d", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2d) {
    TEST_1I("test_maxpool_2d", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2dc) {
    TEST_1I("test_maxpool_2d_ceil", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2dd) {
    TEST_1I("test_maxpool_2d_dilations", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2dp) {
    TEST_1I("test_maxpool_2d_pads", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_3d) {
    TEST_1I("test_maxpool_3d", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool) {
    UnitTest_add(test_maxpool_1d);
    UnitTest_add(test_maxpool_2d);
    UnitTest_add(test_maxpool_2dc);
    // UnitTest_add(test_maxpool_2dd);  // Failed!
    UnitTest_add(test_maxpool_2dp);
    UnitTest_add(test_maxpool_3d);
    return NULL;
}

// ---------------------- Relu   ----------------------

UnitTest_fn_def(test_relu) {
    TEST_1I("test_relu", x, y);
    return NULL;
}

// ---------------------- Reshape ---------------------

UnitTest_fn_def(test_reshape_ar) {
    TEST_2I("test_reshape_allowzero_reordered", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_ed) {
    TEST_2I("test_reshape_extended_dims", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_nd) {
    TEST_2I("test_reshape_negative_dim", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_ned) {
    TEST_2I("test_reshape_negative_extended_dims", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_od) {
    TEST_2I("test_reshape_one_dim", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_rd) {
    TEST_2I("test_reshape_reduced_dims", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape) {
    // UnitTest_add(test_reshape_ar); /** TODO: Segmentation fault */
    UnitTest_add(test_reshape_ed);
    UnitTest_add(test_reshape_nd);
    UnitTest_add(test_reshape_ned);
    UnitTest_add(test_reshape_od);
    UnitTest_add(test_reshape_rd);
    return NULL;
}

// ---------------------- Exit   ----------------------

UnitTest_fn_def(test_node_exit) {
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}


// ---------------------- All    ----------------------

UnitTest_fn_def(test_all) {
    UnitTest_add(test_node_init);

    UnitTest_add(test_abs);
    UnitTest_add(test_add);
    UnitTest_add(test_conv);
    UnitTest_add(test_matmul);
    UnitTest_add(test_maxpool);
    UnitTest_add(test_relu);
    UnitTest_add(test_reshape);

    UnitTest_add(test_node_exit);
    return NULL;
}

UnitTest_run(test_all);