#include "../../evo.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


image_t* image_read_mnist(const char* image_filename, const char* label_filename) {
  FILE* img_file = fopen(image_filename, "rb");
    if (!img_file) {
        fprintf(stderr, "Cannot open image file %s\n", image_filename);
        return NULL;
    }

    FILE* lbl_file = fopen(label_filename, "rb");
    if (!lbl_file) {
        fprintf(stderr, "Cannot open label file %s\n", label_filename);
        fclose(img_file);
        return NULL;
    }

    uint32_t magic, num_images, num_labels, rows, cols;
    size_t read_result;

    read_result = fread(&magic, 4, 1, img_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read magic number from image file\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }
    
    read_result = fread(&num_images, 4, 1, img_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read number of images from image file\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }
    
    read_result = fread(&rows, 4, 1, img_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read number of rows from image file\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }
    
    read_result = fread(&cols, 4, 1, img_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read number of columns from image file\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }

    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    read_result = fread(&magic, 4, 1, lbl_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read magic number from label file\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }

    read_result = fread(&num_labels, 4, 1, lbl_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read number of labels from label file\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }

    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    if (num_images != num_labels) {
        fprintf(stderr, "Number of images and labels do not match\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }

    image_t* image = (image_t*)malloc(sizeof(image_t));
    image->name = strdup(image_filename);
    image->type = IMAGE_TYPE_MNIST;
    image->raw = tensor_new(strdup(image_filename), TENSOR_TYPE_UINT8);
    image->attr_vec = vector_create();

    uint32_t img_size = num_images * rows * cols;
    uint32_t lbl_size = num_labels;

    uint8_t* img_data = (uint8_t*)malloc(img_size);
    uint8_t* lbl_data = (uint8_t*)malloc(lbl_size);

    read_result = fread(img_data, img_size, 1, img_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read image data\n");
        free(img_data);
        free(lbl_data);
        image_free(image);
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }

    read_result = fread(lbl_data, lbl_size, 1, lbl_file);
    if (read_result != 1) {
        fprintf(stderr, "Failed to read label data\n");
        free(img_data);
        free(lbl_data);
        image_free(image);
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }

    fclose(img_file);
    fclose(lbl_file);

    int dims[4] = {
        num_images,
        rows,
        cols,
        1
    };
    tensor_reshape(image->raw, 4, dims);
    tensor_apply(image->raw, img_data, img_size);

    // 将标签数据存储为属性: label bytes
    char key[20];
    sprintf(key, "label");
    attribute_t* attr = attribute_bytes(key, lbl_data, num_labels);
    vector_add(&(image->attr_vec), attr);
    free(lbl_data);

    return image;
}