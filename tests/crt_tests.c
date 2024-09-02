#include "sob.h"
#include <evo.h>

/**
 *  ** Mnist Model ** :
 *      - Flatten           : [1,1,28,28] ->     [1,784]
 *      - FC                :     [1,784] ->     [1,500]
 *      - ReLU              :     [1,500] ->     [1,500]
 *      - FC                :     [1,500] ->      [1,10]
 *      - Softmax           :      [1,10] ->      [1,10]
 *
 *  ** Train Config ** :
 *      - learning_rate     = 0.1(hinton_loss) else 0.35
 *      - optimizer         = Adam
 *      - num_epochs        = 600(hinton_loss) else 60
 *      - batch_size        = 4096
 */
model_t* mnist_model() {
    model_t* mdl = model_new("mnist_model");
    attribute_t* upsample_mode = attribute_string("mode", hashmap_str_lit("bilinear"));
    graph_add_input(mdl->graph, 4, (int[]){1, 1, 28, 28});
    tensor_t* input = model_get_tensor(mdl, "Input0");
    tensor_t* sc = tensor_new_float32("Upsample_scale", (int[]){1, 4}, 2, (float[]){1, 1, 0.5, 0.5}, 4);
    graph_add_layer(mdl->graph, OP_TYPE_UPSAMPLE, (tensor_t*[]){input, sc}, 2, 1, (attribute_t*[]){upsample_mode}, 1);
    graph_add_flatten(mdl->graph);
    graph_add_dense(mdl->graph, 500, "relu");
    graph_add_dense(mdl->graph, 10, "softmax");
    return mdl;
}

UnitTest_fn_def(test_model_create) {
    device_reg("cpu");
    // Dataset
    const char* image_filename = "picture/mnist/t10k-images-idx3-ubyte";
    const char* label_filename = "picture/mnist/t10k-labels-idx1-ubyte";
    image_t* imgs = image_load_mnist(image_filename, label_filename);
    image_t* img_demo = image_get(imgs, 0);
    tensor_t* ts_demo = tensor_cast(img_demo->raw, TENSOR_TYPE_FLOAT32);

    // Model
    model_t* mdl = mnist_model();
    tensor_t * in = model_get_tensor(mdl, "Input0");
    tensor_dump2(ts_demo);
    tensor_copy(in, ts_demo);
    graph_prerun(mdl->graph);
    graph_run(mdl->graph);
    graph_posrun(mdl->graph);
    graph_dump(mdl->graph);
    model_dump_tensor(mdl); 
    tensor_t * out = model_get_tensor(mdl, "Upsample0_out0");
    tensor_dump2(out);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_model_create);
    return NULL;
}

UnitTest_run(test_all);