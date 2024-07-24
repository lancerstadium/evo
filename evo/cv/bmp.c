#include "../evo.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


EVO_PACKED(typedef struct {
    uint16_t bf_type;
    uint32_t bf_size;
    uint16_t bf_reserved1;
    uint16_t bf_reserved2;
    uint32_t bf_off_bits;
}) bmp_file_header_t;

EVO_PACKED(typedef struct {
    uint32_t bi_size;
    int32_t bi_width;
    int32_t bi_height;
    uint16_t bi_planes;
    uint16_t bi_bit_count;
    uint32_t bi_compression;
    uint32_t bi_size_image;
    int32_t bi_x_pels_per_meter;
    int32_t bi_y_pels_per_meter;
    uint32_t bi_clr_used;
    uint32_t bi_clr_important;
}) bmp_info_header_t;


image_t* image_load_bmp(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    bmp_file_header_t file_header;
    bmp_info_header_t info_header;


    if (fread(&file_header, sizeof(bmp_file_header_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read BMP file header\n");
        fclose(file);
        return NULL;
    }
    if (fread(&info_header, sizeof(bmp_info_header_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read BMP info header\n");
        fclose(file);
        return NULL;
    }

    if (file_header.bf_type != 0x4D42) {
        fprintf(stderr, "Not a BMP file %s\n", filename);
        fclose(file);
        return NULL;
    }

    image_t* image = (image_t*)malloc(sizeof(image_t));
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image structure\n");
        fclose(file);
        return NULL;
    }
    image->name = strdup(filename);
    image->type = IMAGE_TYPE_BMP;
    image->raw = tensor_new(strdup(filename), TENSOR_TYPE_UINT8);
    image->attr_vec = vector_create();

    int width = info_header.bi_width;
    int height = info_header.bi_height;
    int bit_count = info_header.bi_bit_count;

    int row_padded = (width * bit_count / 8 + 3) & (~3);
    uint8_t* data = (uint8_t*)malloc(row_padded * height);

    fseek(file, file_header.bf_off_bits, SEEK_SET);
    if (fread(data, row_padded * height, 1, file) != 1) {
        fprintf(stderr, "Failed to read BMP data\n");
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);

    uint8_t* pixel_data = (uint8_t*)malloc(width * height * (bit_count / 8));

    for (int i = 0; i < height; ++i) {
        memcpy(pixel_data + (height - i - 1) * width * (bit_count / 8), data + i * row_padded, width * (bit_count / 8));
    }
    free(data);

    int dims[4] = {
        1,
        (bit_count / 8),
        height,
        width,
    };
    tensor_reshape(image->raw, 4, dims);
    tensor_apply(image->raw, pixel_data, width * height * (bit_count / 8));
    return image;
}