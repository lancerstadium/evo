#include "sob.h"
#include <evo.h>

#define MD(s)           "model/"s"/model.onnx"
#define TI(s, I, i)     "model/"s"/test_data_set_"#I"/input_"#i".pb"
#define TO(s, I, i)     "model/"s"/test_data_set_"#I"/output_"#i".pb"

#define TESTD_1I(s, i, o, I)                      \
    model_t* mdl = sez->load_model(sez, MD(s));   \
    tensor_t* a = model_get_tensor(mdl, i);       \
    tensor_t* b = model_get_tensor(mdl, o);       \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0)); \
    tensor_t* t1 = sez->load_tensor(TO(s, I, 0)); \
    tensor_copy(a, t0);                           \
    graph_prerun(mdl->graph);                     \
    graph_run(mdl->graph);                        \
    graph_dump(mdl->graph);                       \
    tensor_dump2(b);                              \
    tensor_dump2(t1);                             \
    mdl->sez->unload(mdl);

#define TEST_1I(s, i, o, I)                         \
    model_t* mdl = sez->load_model(sez, MD(s));     \
    tensor_t* a = model_get_tensor(mdl, i);         \
    tensor_t* b = model_get_tensor(mdl, o);         \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0));   \
    tensor_t* t1 = sez->load_tensor(TO(s, I, 0));   \
    tensor_copy(a, t0);                             \
    graph_prerun(mdl->graph);                       \
    graph_run(mdl->graph);                          \
    UnitTest_ast(tensor_equal(b, t1), s " failed"); \
    mdl->sez->unload(mdl);

#define TESTD_2I(s, i0, i1, o, I)                 \
    model_t* mdl = sez->load_model(sez, MD(s));   \
    tensor_t* a = model_get_tensor(mdl, i0);      \
    tensor_t* b = model_get_tensor(mdl, i1);      \
    tensor_t* c = model_get_tensor(mdl, o);       \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0)); \
    tensor_t* t1 = sez->load_tensor(TI(s, I, 1)); \
    tensor_t* t2 = sez->load_tensor(TO(s, I, 0)); \
    tensor_copy(a, t0);                           \
    tensor_copy(b, t1);                           \
    graph_prerun(mdl->graph);                     \
    graph_run(mdl->graph);                        \
    tensor_dump2(c);                              \
    tensor_dump2(t2);                             \
    mdl->sez->unload(mdl);

#define TEST_2I(s, i0, i1, o, I)                    \
    model_t* mdl = sez->load_model(sez, MD(s));     \
    tensor_t* a = model_get_tensor(mdl, i0);        \
    tensor_t* b = model_get_tensor(mdl, i1);        \
    tensor_t* c = model_get_tensor(mdl, o);         \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0));   \
    tensor_t* t1 = sez->load_tensor(TI(s, I, 1));   \
    tensor_t* t2 = sez->load_tensor(TO(s, I, 0));   \
    tensor_copy(a, t0);                             \
    tensor_copy(b, t1);                             \
    graph_prerun(mdl->graph);                       \
    graph_run(mdl->graph);                          \
    UnitTest_ast(tensor_equal(c, t2), s " failed"); \
    mdl->sez->unload(mdl);

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
    TEST_1I("mobilenet_v2_7", "data", "mobilenetv20_output_flatten0_reshape0", 0); // 155
    return NULL;
}

// ---------------------- ShuffleNet -----------------

UnitTest_fn_def(test_shufflenet_v1_9) {
    // TESTD_1I("shufflenet_v1_9", "gpu_0/data_0", "gpu_0/softmax_1", 0);       // 203 Op: Transpose Sum Gemm Softmax
    return NULL;
}

// ---------------------- SqueezeNet -----------------

UnitTest_fn_def(test_squeezenet_v11_7) {
    TESTD_1I("squeezenet_v11_7", "data", "squeezenet0_flatten0_reshape0", 0);   // 66 Op: Concat Dropout AveragePool
    return NULL;
}

// ---------------------- TinyYolo -------------------

UnitTest_fn_def(test_tinyyolo_v2_8) {
    // TESTD_1I("tinyyolo_v2_8", "image", "grid", 0);                              // 33: Segment fault
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
    UnitTest_add(test_squeezenet_v11_7);
    UnitTest_add(test_tinyyolo_v2_8);
    UnitTest_add(test_model_exit);
    return NULL;
}

UnitTest_run(test_all);