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
        image_t* new_img = image_get_batch(img, 3, (int[]){3, 4, 6});
        image_dump_raw(new_img, -1);
        // tensor_dump2(new_img->raw);
        image_free(img);
    } else {
        fprintf(stderr, "Failed to read img\n");
    }

    return NULL;
}

UnitTest_fn_def(test_read_png) {
    const char* img_path = "picture/basn0g04.png";
    image_t* img = image_load(img_path);
    // image_dump_raw(img, 0);
    image_save(img, "demo.png");
    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_read_mnist);
    UnitTest_add(test_read_png);
    return NULL;
}

UnitTest_run(test_all);