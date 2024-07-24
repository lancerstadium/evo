#include "../evo.h"
#include "../util/log.h"
#include <string.h>
#include <stdio.h>

attribute_t* image_get_attr(image_t* img, const char* name) {
    attribute_t* attr;
    int i;
    if (img && name) {
        for (i = 0; i < vector_size(img->attr_vec); i++) {
            attr = img->attr_vec[i];
            if (strcmp(attr->name, name) == 0) {
                return attr;
            }
        }
    }
    return NULL;
}

void image_dump_shape(image_t* img) {
    if (img && img->raw) {
        LOG_INFO("%s", tensor_dump_shape(img->raw));
    }
}

void image_dump_raw(image_t* img, int i) {
    if(!img || !img->raw || !img->raw->datas) return;
    uint8_t* data = (uint8_t*)img->raw->datas;
    if(i >= 0) {
        for (int r = 0; r < img->raw->dims[2]; ++r) {
            for (int c = 0; c < img->raw->dims[2]; ++c) {
                printf("%3d ", data[i * img->raw->dims[2] * img->raw->dims[3] + r * img->raw->dims[3] + c]);
            }
            printf("\n");
        }
    } else {
        for(int j = 0; j < img->raw->dims[0] || j < 20; ++j) {
            image_dump_raw(img, j);
        }
    }
}

void image_free(image_t* img) {
    if (img) {
        if (img->name) free(img->name);
        if (img->raw) tensor_free(img->raw);
        if (img->attr_vec) vector_free(img->attr_vec);
        free(img);
        img = NULL;
    }
}