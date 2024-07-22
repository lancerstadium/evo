#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_read_mnist) {
    const char* image_filename = "model/mnist_8/test_data_set_3/t10k-images-idx3-ubyte";
    const char* label_filename = "model/mnist_8/test_data_set_3/t10k-labels-idx1-ubyte";

    image_t* img = image_read_mnist(image_filename, label_filename);
    if (img) {
        printf("Image name: %s\n", img->name);
        printf("Image type: %d\n", img->type);
        printf("Image dimensions: %d x %d x %d x %d\n", img->raw->dims[0], img->raw->dims[1], img->raw->dims[2], img->raw->dims[3]);

        // 打印部分数据以验证
        uint8_t* data = (uint8_t*)img->raw->datas;
        attribute_t* attr = image_get_attr(img, "label");
        for (int i = 0; i < 10; ++i) {
            printf("Label[%2d]: %d\n", i, attr->bs[i]);
            for (int r = 0; r < img->raw->dims[1]; ++r) {
                for (int c = 0; c < img->raw->dims[2]; ++c) {
                    printf("%3d ", data[i * img->raw->dims[1] * img->raw->dims[2] + r * img->raw->dims[2] + c]);
                }
                printf("\n");
            }
            printf("\n");
        }
        image_free(img);
    } else {
        fprintf(stderr, "Failed to read img\n");
    }

    return NULL;
}

UnitTest_fn_def(test_all) {
    UnitTest_add(test_read_mnist);
    return NULL;
}

UnitTest_run(test_all);