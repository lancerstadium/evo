#include "sob.h"
#include <evo.h>

#define MD(s)           "model/"s"/model.onnx"
#define TI(s, I, i)     "model/"s"/test_data_set_"#I"/input_"#i".pb"
#define TO(s, I, i)     "model/"s"/test_data_set_"#I"/output_"#i".pb"

#define TESTD_1I(s, i, o, I)                      \
    context_t* ctx = sez->load_model(sez, MD(s)); \
    tensor_t* a = context_get_tensor(ctx, i);     \
    tensor_t* b = context_get_tensor(ctx, o);     \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0)); \
    tensor_t* t1 = sez->load_tensor(TO(s, I, 0)); \
    tensor_copy(a, t0);                           \
    graph_prerun(ctx->graph);                     \
    graph_run(ctx->graph);                        \
    graph_exec_report(ctx->graph);                \
    tensor_dump2(b);                              \
    tensor_dump2(t1);                             \
    ctx->sez->unload(ctx);

#define TEST_1I(s, i, o, I)                         \
    context_t* ctx = sez->load_model(sez, MD(s));   \
    tensor_t* a = context_get_tensor(ctx, i);       \
    tensor_t* b = context_get_tensor(ctx, o);       \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0));   \
    tensor_t* t1 = sez->load_tensor(TO(s, I, 0));   \
    tensor_copy(a, t0);                             \
    graph_prerun(ctx->graph);                       \
    graph_run(ctx->graph);                          \
    UnitTest_ast(tensor_equal(b, t1), s " failed"); \
    ctx->sez->unload(ctx);

#define TESTD_2I(s, i0, i1, o, I)                 \
    context_t* ctx = sez->load_model(sez, MD(s)); \
    tensor_t* a = context_get_tensor(ctx, i0);    \
    tensor_t* b = context_get_tensor(ctx, i1);    \
    tensor_t* c = context_get_tensor(ctx, o);     \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0)); \
    tensor_t* t1 = sez->load_tensor(TI(s, I, 1)); \
    tensor_t* t2 = sez->load_tensor(TO(s, I, 0)); \
    tensor_copy(a, t0);                           \
    tensor_copy(b, t1);                           \
    graph_prerun(ctx->graph);                     \
    graph_run(ctx->graph);                        \
    tensor_dump2(c);                              \
    tensor_dump2(t2);                             \
    ctx->sez->unload(ctx);

#define TEST_2I(s, i0, i1, o, I)                    \
    context_t* ctx = sez->load_model(sez, MD(s));   \
    tensor_t* a = context_get_tensor(ctx, i0);      \
    tensor_t* b = context_get_tensor(ctx, i1);      \
    tensor_t* c = context_get_tensor(ctx, o);       \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0));   \
    tensor_t* t1 = sez->load_tensor(TI(s, I, 1));   \
    tensor_t* t2 = sez->load_tensor(TO(s, I, 0));   \
    tensor_copy(a, t0);                             \
    tensor_copy(b, t1);                             \
    graph_prerun(ctx->graph);                       \
    graph_run(ctx->graph);                          \
    UnitTest_ast(tensor_equal(c, t2), s " failed"); \
    ctx->sez->unload(ctx);

serializer_t * sez;

// ---------------------- Init  ----------------------

UnitTest_fn_def(test_model_init) {
    device_reg("cpu");
    sez = serializer_new("onnx");
    return NULL;
}

// ---------------------- Mnist ----------------------

UnitTest_fn_def(test_mnist_8) {
    TEST_1I("mnist_8", "Input3", "Plus214_Output_0", 0);
    return NULL;
}

// ---------------------- MobileNet ------------------

UnitTest_fn_def(test_mobilenet_v2_7) {
    // TESTD_1I("mobilenet_v2_7", "data", "mobilenetv20_output_flatten0_reshape0", 0);
    return NULL;
}

// ---------------------- ShuffleNet -----------------

UnitTest_fn_def(test_shufflenet_v1_9) {
    TESTD_1I("shufflenet_v1_9", "gpu_0/data_0", "gpu_0/softmax_1", 0);
    return NULL;
}


// ---------------------- Exit  ----------------------

UnitTest_fn_def(test_model_exit) {
    serializer_free(sez);
    device_unreg("cpu");
}

// ---------------------- All    ----------------------

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_init);
    UnitTest_add(test_mnist_8);
    UnitTest_add(test_mobilenet_v2_7);
    UnitTest_add(test_shufflenet_v1_9);
    UnitTest_add(test_model_exit);
    return NULL;
}

UnitTest_run(test_all);