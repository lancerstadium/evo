#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_read_mnist) {
    const char* image_filename = "model/mnist_8/test_data_set_3/t10k-images-idx3-ubyte";
    const char* label_filename = "model/mnist_8/test_data_set_3/t10k-labels-idx1-ubyte";

    image_t* img = image_load_mnist(image_filename, label_filename);
    if (img) {
        printf("Image name: %s\n", img->name);
        printf("Image type: %d\n", img->type);
        printf("Image dimensions: %d x %d x %d x %d\n", img->raw->dims[0], img->raw->dims[1], img->raw->dims[2], img->raw->dims[3]);

        attribute_t* attr = image_get_attr(img, "label");
        image_t* new_img = image_get_batch(img, 3, (int[]){9, 4, 7});
        image_dump_raw(new_img, 0);
        // image_save(new_img, "1.png");
        image_free(img);
    } else {
        fprintf(stderr, "Failed to read img\n");
    }

    return NULL;
}

UnitTest_fn_def(test_read_img) {
    image_t* img = image_load("picture/p.png");
    UnitTest_msg("%s", image_dump_shape(img));
    // image_dump_raw(img, 0);
    return NULL;
}

UnitTest_fn_def(test_heat_img) {
    float heat[40] = {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.9, 0.8,
        0.7, 0.6, 0.5, 0.3,
        0.3, 0.2, 0.5, 0.7,

        0.2, 0.2, 0.3, 0.4,
        0.2, 0.8, 0.3, 0.5,
        0.2, 0.3, 1.0, 0.5,
        0.2, 0.4, 0.9, 0.5,
        0.2, 0.4, 0.6, 0.5,
    };
    tensor_t *ts = tensor_new_float32("heat", (int[]){1, 2, 5, 4}, 4, heat, 40);
    tensor_t *ts_p = tensor_permute(ts, 4, (int[]){0, 2, 3, 1});
    tensor_dump2(ts_p);
    image_t* heat_img = image_heatmap(ts, 1);
    image_set_deloys(heat_img, (int64_t[]){10, 10}, 2);
    image_save(heat_img, "heat.gif");
    return NULL;
}

UnitTest_fn_def(test_read_png) {
    const char* img_path = "model/lut3d_96/test_data_set_0/input_0.jpg";
    image_t* img = image_load(img_path);
    // image_dump_raw(img, 0);
    UnitTest_msg("%s", image_dump_shape(img));
    image_save(img, "demo.jpg");
    image_t* r_img = image_channel(img, 0);
    UnitTest_msg("%s", image_dump_shape(r_img));
    image_save(r_img, "demo_r.jpg");
    image_save_grey(img, "demo_g.jpg", 0);
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_read_mnist);
    UnitTest_add(test_read_img);
    UnitTest_add(test_heat_img);
    UnitTest_add(test_read_png);
    return NULL;
}

UnitTest_run(test_all);