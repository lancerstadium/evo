#include "sob.h"
#include <evo.h>

#define MD(s)           "model/"s"/model.onnx"
#define TI(s, I, i)     "model/"s"/test_data_set_"#I"/input_"#i".pb"
#define TO(s, I, i)     "model/"s"/test_data_set_"#I"/output_"#i".pb"

// ---------------------- ImageNet Pre ---------------
tensor_t* imagenet_preprocess(image_t* img) {
    if(!img) return NULL;
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

void imagenet_load_label(const char* path, char* labels[1000]) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", path);
        labels = NULL;
        return;
    }
    
    int nlabel = 1000;
    
    for (int i = 0; i < nlabel; i++) {
        labels[i] = malloc(100 * sizeof(char));
        if (labels[i] == NULL) {
            printf("Error allocating memory.\n");
            fclose(file);
            for (int j = 0; j < i; j++) {
                free(labels[j]);
            }
            return;
        }
    }
    
    int i = 0;
    while (i < nlabel && fgets(labels[i], 100, file) != NULL) {
        size_t len = strlen(labels[i]);
        if (len > 0 && labels[i][len - 1] == '\n') {
            labels[i][len - 1] = '\0';  // Remove the newline character at the end
        }
        i++;
    }
    
    fclose(file);
    
    // Printing the labels
    // for (int j = 0; j < nlabel; j++) {
    //     printf("%s\n", labels[j]);
    // }
}

void imagenet_unload_label(char* labels[1000]) {
    if(!labels) return;
    int nlabel = 1000;
    // Freeing allocated memory
    for (int j = 0; j < nlabel; j++) {
        if(labels[j]) free(labels[j]);
    }
    labels = NULL;
}

image_t* imagenet_recover(tensor_t* ts) {
    if(!ts || ts->type != TENSOR_TYPE_FLOAT32) return NULL;
    float* data = ts->datas;
    for(int i = 0; i < ts->ndata; i += 1) {
        data[i] = data[i] * 255;
    }
    image_t* img = image_from_tensor(ts);
    return img;
}

void imagenet_postprocess(tensor_t* out) {
    if(!out) return;
    char* labels[1000];
    imagenet_load_label("picture/imagenet/imagenet.shortnames.list", labels);
    if(!labels) {
        return;
    }
    tensor_t* scores_tmp = tensor_softmax(out, -1);
    tensor_t* scores = tensor_squeeze(scores_tmp, NULL, 0);
    tensor_t* scores_max = tensor_argmax(scores, 0, 1, 0);
    // tensor_dump2(scores_tmp);
    int64_t* scores_idx = scores_max->datas;
    float* scores_res = scores->datas;
    UnitTest_msg("Class: %.2f (%s)", scores_res[scores_idx[0]], labels[scores_idx[0]]);
    imagenet_unload_label(labels);
}


// ---------------------- MobileNet ------------------

UnitTest_fn_def(test_mobilenet_v2_7) {

    // 1. Pre Process
    image_t* img = image_load("picture/kitten.jpg");
    int width = img->raw->dims[2];
    int height = img->raw->dims[1];
    image_save(img, "mobilenet_origin.jpg");
    tensor_t* ts_pre = imagenet_preprocess(img);
    runtime_t* rt = runtime_new("onnx");

    // 2. Model Inference
    runtime_load(rt, MD("mobilenet_v2_7"));
    tensor_dump(ts_pre);
    tensor_t* input = ts_pre;
    // tensor_t* input = runtime_load_tensor(rt, TI("mobilenet_v2_7", 0, 0));
    // tensor_t* output_ref = runtime_load_tensor(rt, TO("mobilenet_v2_7", 0, 0));
    image_t* img_input = imagenet_recover(input);
    image_save(img_input, "mobilenet_input.jpg");
    runtime_set_tensor(rt, "data", input);
    runtime_run(rt);

    // 3. Post Process
    tensor_t* output = runtime_get_tensor(rt, "mobilenetv20_output_flatten0_reshape0");
    // tensor_dump2(output);
    // UnitTest_ast(tensor_equal(output, output_ref), "mobilenet_v2_7 failed");
    imagenet_postprocess(output);


    tensor_t * feature_map = runtime_get_tensor(rt, "mobilenetv20_features_conv1_fwd");
    const char* relu_layer[] = {
        "mobilenetv20_features_linearbottleneck9_relu0_fwd",
        "mobilenetv20_features_linearbottleneck9_relu1_fwd",
        "mobilenetv20_features_linearbottleneck10_relu0_fwd",
        "mobilenetv20_features_linearbottleneck10_relu1_fwd",
        "mobilenetv20_features_linearbottleneck11_relu0_fwd",
        "mobilenetv20_features_linearbottleneck11_relu1_fwd"
    };
    int channel = 2;
    for(int i = 0; i < sizeof(relu_layer)/sizeof(relu_layer[0]); i++) {
        char path[64];
        image_t* origin_img = image_load("picture/kitten.jpg");
        sprintf(path, "heatmap_mobilenet_relu_%d_c%d.png", i, channel);
        tensor_t * ts = runtime_get_tensor(rt, relu_layer[i]);
        image_t* heat_map = image_heatmap(ts, channel);
        image_resize(heat_map, width, height);
        if(i == 0) tensor_dump(heat_map->raw);
        image_t* res_img = image_merge(origin_img, heat_map, 0.3);
        image_save(res_img, path);
    }

    runtime_free(rt);

    return NULL;
}

// ---------------------- SqueezeNet -----------------

UnitTest_fn_def(test_squeezenet_v11_7) {

    // 1. Pre Process
    image_t* img = image_load("picture/kitten.jpg");
    image_save(img, "squeezenet_origin.jpg");
    tensor_t* ts_pre = imagenet_preprocess(img);
    runtime_t* rt = runtime_new("onnx");

    // 2. Model Inference
    runtime_load(rt, MD("squeezenet_v11_7"));
    tensor_t* input = ts_pre;
    // tensor_t* input = runtime_load_tensor(rt, TI("squeezenet_v11_7", 0, 0));
    // tensor_t* output_ref = runtime_load_tensor(rt, TO("squeezenet_v11_7", 0, 0));
    image_t* img_input = imagenet_recover(input);
    image_save(img_input, "squeezenet_input.jpg");
    runtime_set_tensor(rt, "data", input);
    runtime_run(rt);
    graph_dump1(rt->mdl->graph);

    // 3. Post Process
    tensor_t* output = runtime_get_tensor(rt, "squeezenet0_flatten0_reshape0");
    // UnitTest_ast(tensor_equal(output, output_ref), "squeezenet_v11_7 failed");
    imagenet_postprocess(output);
    runtime_free(rt);

    return NULL;
}

// ---------------------- SqueezeNet -----------------

UnitTest_fn_def(test_resnet_18_v1_7) {

    // 1. Pre Process
    image_t* img = image_load("picture/kitten.jpg");
    image_save(img, "resnet_18_origin.jpg");
    tensor_t* ts_pre = imagenet_preprocess(img);
    runtime_t* rt = runtime_new("onnx");

    // 2. Model Inference
    runtime_load(rt, MD("resnet_18_v1_7"));
    tensor_t* input = ts_pre;
    // tensor_t* input = runtime_load_tensor(rt, TI("resnet_18_v1_7", 0, 0));
    // tensor_t* output_ref = runtime_load_tensor(rt, TO("resnet_18_v1_7", 0, 0));
    image_t* img_input = imagenet_recover(input);
    image_save(img_input, "resnet_18_input.jpg");
    runtime_set_tensor(rt, "data", input);
    runtime_run(rt);

    // 3. Post Process
    tensor_t* output = runtime_get_tensor(rt, "resnetv15_dense0_fwd");
    // UnitTest_ast(tensor_equal(output, output_ref), "resnet_18_v1_7 failed");
    imagenet_postprocess(output);
    runtime_free(rt);

    return NULL;
}

// ---------------------- All    ----------------------

UnitTest_fn_def(test_all) {
    // UnitTest_add(test_mobilenet_v2_7);
    UnitTest_add(test_squeezenet_v11_7);
    // UnitTest_add(test_resnet_18_v1_7);
    return NULL;
}

UnitTest_run(test_all);