#include "sob.h"
#include <evo.h>

#define MD(s)           "model/"s"/model.onnx"
#define TI(s, I, i)     "model/"s"/test_data_set_"#I"/input_"#i".pb"
#define TO(s, I, i)     "model/"s"/test_data_set_"#I"/output_"#i".pb"

// ---------------------- ImageNet Pre ---------------
tensor_t* imagenet_preprocess(image_t* img) {
    image_resize(img, 256, 256);
    image_crop_center(img, 224, 224);
    tensor_t* ts = tensor_nhwc2nchw(img->raw);
    // normalize
    float mean_vec[3] = {0.485, 0.456, 0.406};
    float stddev_vec[3] = {0.229, 0.224, 0.225};
    int ndata = ts->ndata;
    float datas[ndata];
    uint8_t* data = ts->datas;
    for(int i = 0; i < ts->ndata; i += 3) {
        for(int j = 0; j < 3; j++) {
            datas[i + j] = ((float)data[i + j]/255 - mean_vec[j]) / stddev_vec[j];
            // datas[i + j] = (float)data[i + j]/255;
        }
    }
    tensor_t* new_ts = tensor_new_float32(ts->name, (int[]){1, 3, 224, 224}, 4, datas, ndata);
    tensor_free(ts);
    return new_ts;
}

image_t* imagenet_recover(tensor_t* ts) {
    if(ts->type != TENSOR_TYPE_FLOAT32) return NULL;
    float* data = ts->datas;
    for(int i = 0; i < ts->ndata; i += 1) {
        data[i] = data[i] * 255;
    }
    image_t* img = image_from_tensor(ts);
    return img;
}

// ---------------------- MobileNet ------------------

UnitTest_fn_def(test_mobilenet_v2_7) {

    // 1. Pre Process
    image_t* img = image_load("picture/kitten.jpg");
    image_save(img, "mobilenet_origin.jpg");
    tensor_t* ts_pre = imagenet_preprocess(img);
    image_t* img_pre = imagenet_recover(ts_pre);
    image_save(img_pre, "mobilenet_preprocess.jpg");
    runtime_t* rt = runtime_new("onnx");

    // 2. Model Inference
    runtime_load(rt, MD("mobilenet_v2_7"));
    tensor_t* input = runtime_load_tensor(rt, TI("mobilenet_v2_7", 0, 0));
    tensor_t* output_ref = runtime_load_tensor(rt, TO("mobilenet_v2_7", 0, 0));
    // tensor_t* input = ts_pre;
    runtime_set_tensor(rt, "data", input);
    runtime_run(rt);

    // const char* relu_layer[] = {
    //     "mobilenetv20_features_linearbottleneck9_relu0_fwd",
    //     "mobilenetv20_features_linearbottleneck9_relu1_fwd",
    //     "mobilenetv20_features_linearbottleneck10_relu0_fwd",
    //     "mobilenetv20_features_linearbottleneck10_relu1_fwd",
    //     "mobilenetv20_features_linearbottleneck11_relu0_fwd",
    //     "mobilenetv20_features_linearbottleneck11_relu1_fwd"
    // };
    // int channel = 2;
    // for(int i = 0; i < sizeof(relu_layer)/sizeof(relu_layer[0]); i++) {
    //     char path[64];
    //     sprintf(path, "heatmap_mobiilenet_relu_%d_c%d.png", i, channel);
    //     tensor_t * ts = runtime_get_tensor(rt, relu_layer[i]);
    //     image_t *heat_map = image_heatmap(ts, channel);
    //     tensor_dump(ts);
    //     image_save(heat_map, path);
    // }

    // 3. Post Process
    tensor_t* output = runtime_get_tensor(rt, "mobilenetv20_output_flatten0_reshape0");
    // UnitTest_ast(tensor_equal(output, output_ref), "mobilenet_v2_7 failed");
    tensor_t* output_prob_sm = tensor_softmax(output, 0);
    tensor_t* output_prob_sq = tensor_squeeze(output_prob_sm, NULL, 0);
    tensor_dump2(output_prob_sq);
    runtime_free(rt);

    return NULL;
}

// ---------------------- All    ----------------------

UnitTest_fn_def(test_all) {
    UnitTest_add(test_mobilenet_v2_7);
    return NULL;
}

UnitTest_run(test_all);