#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CIFAR10_IMG_WIDTH   32
#define CIFAR10_IMG_HEIGHT  32
#define CIFAR10_IMG_CHANNEL 3
#define CIFAR10_IMG_BATCH   10000
#define CIFAR10_IMG_SIZE    (CIFAR10_IMG_WIDTH * CIFAR10_IMG_HEIGHT * CIFAR10_IMG_CHANNEL)


image_t* image_load_cifar10(const char* image_filename, int image_number) {
    FILE* img_file = fopen(image_filename, "rb");
    if (!img_file) {
        LOG_INFO("Cannot open image file %s\n", image_filename);
        return NULL;
    }
    int num;
    if(image_number < 0) {
        num = CIFAR10_IMG_BATCH;
    } else {
        num = MIN(image_number, CIFAR10_IMG_BATCH);
    }
    unsigned char* data = malloc(num * CIFAR10_IMG_SIZE);
    int64_t* label = malloc(num * sizeof(int64_t));
    unsigned char buffer[1 + CIFAR10_IMG_SIZE];
    for (int i = 0; i < num; i++) {
        fread(buffer, sizeof(unsigned char), 1 + CIFAR10_IMG_SIZE, img_file);
        label[i] = buffer[0];
        memcpy(data + i * CIFAR10_IMG_SIZE, buffer + 1, CIFAR10_IMG_SIZE);
    }
    
    image_t* image = (image_t*)malloc(sizeof(image_t));
    image->name = strdup(image_filename);
    image->type = IMAGE_TYPE_UNKNOWN;
    image->raw = tensor_new(strdup(image_filename), TENSOR_TYPE_UINT8);
    image->attr_vec = vector_create();
    tensor_reshape(image->raw, 4, (int[]){num, CIFAR10_IMG_CHANNEL, CIFAR10_IMG_HEIGHT, CIFAR10_IMG_WIDTH});
    tensor_apply(image->raw, data, num * CIFAR10_IMG_SIZE);

    attribute_t* attr = attribute_ints("label", label, num);
    vector_add(&(image->attr_vec), attr);
    free(data);
    free(label);
    fclose(img_file);
    return image;
}