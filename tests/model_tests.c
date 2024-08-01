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
    tensor_dump2(b);                              \
    tensor_dump2(t1);                             \
    graph_dump(mdl->graph);                       \
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

#define TEST_1I_BEGIN(s, i, o, I)                 \
    model_t* mdl = sez->load_model(sez, MD(s));   \
    tensor_t* a = model_get_tensor(mdl, i);       \
    tensor_t* b = model_get_tensor(mdl, o);       \
    tensor_t* t0 = sez->load_tensor(TI(s, I, 0)); \
    tensor_t* t1 = sez->load_tensor(TO(s, I, 0)); \
    tensor_copy(a, t0);                           \
    graph_prerun(mdl->graph);                     \
    graph_run(mdl->graph);                        \
    UnitTest_ast(tensor_equal(b, t1), s " failed");

#define TEST_NORM_END() \
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
    TEST_1I_BEGIN("mnist_8", "Input3", "Plus214_Output_0", 0);

    const char* relu_layer[] = {
        "ReLU114_Output_0"
    };

    // image_t *origin_input = image_from_tensor(a);
    // image_save(origin_input, "mnist_input.png");
    // int channel = 0;
    // for(int i = 0; i < sizeof(relu_layer)/sizeof(relu_layer[0]); i++) {
    //     char path[32];
    //     sprintf(path, "heatmap_mnist_relu_%d_c%d.png", i, channel);
    //     tensor_t * ts = model_get_tensor(mdl, relu_layer[i]);
    //     image_t *grey_map = image_from_tensor(ts);
    //     tensor_dump(ts);
    //     image_save_grey(grey_map, path, channel);
    // }

    TEST_NORM_END();
    return NULL;
}

// ---------------------- MobileNet ------------------

UnitTest_fn_def(test_mobilenet_v2_7) {
    TEST_1I_BEGIN("mobilenet_v2_7", "data", "mobilenetv20_output_flatten0_reshape0", 0); // 66

    // const char* relu_layer[] = {
    //     "mobilenetv20_features_linearbottleneck9_relu0_fwd",
    //     "mobilenetv20_features_linearbottleneck9_relu1_fwd",
    //     "mobilenetv20_features_linearbottleneck10_relu0_fwd",
    //     "mobilenetv20_features_linearbottleneck10_relu1_fwd",
    //     "mobilenetv20_features_linearbottleneck11_relu0_fwd",
    //     "mobilenetv20_features_linearbottleneck11_relu1_fwd"
    // };

    // image_t *origin_input = image_from_tensor(a);
    // image_save(origin_input, "origin_input.png");
    // int channel = 2;
    // for(int i = 0; i < sizeof(relu_layer)/sizeof(relu_layer[0]); i++) {
    //     char path[32];
    //     sprintf(path, "heatmap_mobiilenet_relu_%d_c%d.png", i, channel);
    //     tensor_t * ts = model_get_tensor(mdl, relu_layer[i]);
    //     image_t *heat_map = image_heatmap(ts, channel);
    //     tensor_dump(ts);
    //     image_save(heat_map, path);
    // }

    TEST_NORM_END();
    return NULL;
}

// ---------------------- ShuffleNet -----------------

UnitTest_fn_def(test_shufflenet_v1_9) {
    // TEST_1I("shufflenet_v1_9", "gpu_0/data_0", "gpu_0/softmax_1", 0);       // 203 Op: Transpose Sum Gemm Softmax
    return NULL;
}

// ---------------------- SqueezeNet -----------------

UnitTest_fn_def(test_squeezenet_v11_7) {
    // TEST_1I("squeezenet_v11_7", "data", "squeezenet0_flatten0_reshape0", 0);   // 66 Op: Concat Dropout AveragePool
    return NULL;
}

// ---------------------- TinyYolo -------------------

UnitTest_fn_def(test_tinyyolo_v2_8) {
    TEST_1I_BEGIN("tinyyolo_v2_8", "image", "grid", 0);                              // 33

    // for(int i = 1; i <= 8; i++) {
    //     char name[32];
    //     char path[32];
    //     sprintf(name, "leakyrelu_%d_output", i);
    //     sprintf(path, "heat_map_leakyrelu_%d.jpg", i);
    //     tensor_t * ts = model_get_tensor(mdl, name);
    //     image_t *heat_map = image_heatmap(ts, 0);
    //     image_save(heat_map, path);
    // }
    tensor_t* in_img = tensor_permute(t0, 4, (int[]){0, 1, 3, 2});
    image_t *img = image_from_tensor(in_img);
    canvas_t *cav = canvas_from_image(img);
    // canvas_export(cav, "tinyyolo.jpg");
    UnitTest_msg("%s", image_dump_shape(cav->background));

    TEST_NORM_END();
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