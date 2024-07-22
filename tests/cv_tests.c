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

        // 打印部分数据以验证
        uint8_t* data = (uint8_t*)img->raw->datas;
        attribute_t* attr = image_get_attr(img, "label");
        image_dump_raw(img, 1);
        image_dump_raw(img, 3);
        image_dump_raw(img, 5);
        image_free(img);
    } else {
        fprintf(stderr, "Failed to read img\n");
    }

    return NULL;
}

UnitTest_fn_def(test_read_bmp) {
    const char* img_path = "picture/color.bmp";
    image_t* img = image_load_bmp(img_path);
    image_dump_raw(img, 1);
    UnitTest_msg("%s", tensor_dump_shape(img->raw));
    return NULL;
}

UnitTest_fn_def(test_read_jpg) {
    const char* img_path = "model/lut3d_96/test_data_set_0/input_0.jpg";
    image_t* img = image_load_jpg(img_path);
    tensor_dump(img->raw);
    return NULL;
}

UnitTest_fn_def(test_all) {
    // UnitTest_add(test_read_mnist);
    UnitTest_add(test_read_bmp);
    return NULL;
}

UnitTest_run(test_all);