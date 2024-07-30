#include <string.h>
#include <stdio.h>
#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

char* image_dump_shape(image_t* img) {
    if (img && img->raw) {
        return tensor_dump_shape(img->raw);
    }
    return NULL;
}

void image_dump_raw(image_t* img, int i) {
    if(!img || !img->raw || !img->raw->datas) return;
    uint8_t* data = (uint8_t*)img->raw->datas;
    if(i >= 0) {
        for (int s = 0; s < img->raw->dims[1]; ++s) {
            for (int r = 0; r < img->raw->dims[2]; ++r) {
                for (int c = 0; c < img->raw->dims[3]; ++c) {
                    printf("%3d ", data[i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] + s * img->raw->dims[2] * img->raw->dims[3] + r * img->raw->dims[3] + c]);
                }
                printf("\n");
            }
            printf("\n");
        }
    } else {
        for(int j = 0; j < img->raw->dims[0] && j < 20; ++j) {
            image_dump_raw(img, j);
        }
    }
}
tensor_t* image_get_raw(image_t *img, int i) {
    if(img && img->raw && i >= 0 && i < img->raw->dims[0]) {
        tensor_t * ts = tensor_new(sys_strdup(img->name), TENSOR_TYPE_UINT8);
        int dims[4] = {
            1,
            img->raw->dims[1],
            img->raw->dims[2],
            img->raw->dims[3]
        };
        tensor_reshape(ts, 4, dims);
        tensor_apply(ts, (void*)(img->raw->datas + i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3]), img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3]);
        return ts;
    }
    return img->raw;
}

tensor_t* image_get_raw_batch(image_t *img, int n, int *idx) {
    if(img && img->raw) {
        tensor_t * ts = tensor_new(img->name, TENSOR_TYPE_UINT8);
        int dims[4] = {
            n,
            img->raw->dims[1],
            img->raw->dims[2],
            img->raw->dims[3]
        };
        tensor_reshape(ts, 4, dims);
        void* data = malloc(img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] * n);
        for(int i = 0; i < n; ++i) {
            if( idx[i] >= 0 && idx[i] < img->raw->dims[0]) {
                memcpy(data + i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3], 
                    img->raw->datas + idx[i] * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3], 
                    img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3]);
            }
        }
        tensor_apply(ts, data, img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] * n);
        return ts;
    }
    return img->raw;
}

image_t* image_get(image_t* img, int i) {
    if(img && img->raw && i >= 0) {
        image_t* new_img = (image_t*)malloc(sizeof(image_t));
        new_img->name = sys_strdup(img->name);
        new_img->attr_vec = vector_create();
        new_img->type = img->type;
        new_img->raw = image_get_raw(img, i);
        return new_img;
    }
    return NULL;
}

image_t* image_get_batch(image_t* img, int n, int *idx) {
    if(img && img->raw) {
        image_t* new_img = (image_t*)malloc(sizeof(image_t));
        new_img->name = sys_strdup(img->name);
        new_img->attr_vec = vector_create();
        new_img->type = img->type;
        new_img->raw = image_get_raw_batch(img, n, idx);
        return new_img;
    }
    return NULL;
}

image_type_t image_get_type(const char* name) { 
    char* ext = sys_get_file_ext(name);
    if(strcmp(ext, "bmp") == 0) {
        return IMAGE_TYPE_BMP;
    } else if(strcmp(ext, "jpg") == 0) {
        return IMAGE_TYPE_JPG;
    } else if(strcmp(ext, "png") == 0) {
        return IMAGE_TYPE_PNG;
    } else if(strcmp(ext, "tga") == 0) {
        return IMAGE_TYPE_TGA;
    } else if(strcmp(ext, "hdr") == 0) {
        return IMAGE_TYPE_HDR;
    } else {
        return IMAGE_TYPE_UNKNOWN;
    }
}

image_t* image_blank(const char* name, size_t height, size_t width) {
    image_t* img = (image_t*)malloc(sizeof(image_t));
    if(img) {
        img->name = sys_strdup(name);
        img->attr_vec = vector_create();
        img->type = IMAGE_TYPE_UNKNOWN;
        img->raw = tensor_new(sys_strdup(name), TENSOR_TYPE_UINT8);
        tensor_reshape(img->raw, 4, (int[]){1, 4, height, width});
        uint8_t * data = sys_malloc(4 * height * width * sizeof(uint8_t));
        tensor_apply(img->raw, (void*)data, 4 * height * width);
        free(data);
        data = NULL;
        return img;
    }
    return NULL;
}

image_t* image_load(const char* name) {
    image_t* img = (image_t*)malloc(sizeof(image_t));
    if(img) {
        img->name = sys_strdup(name);
        img->attr_vec = vector_create();
        img->type = image_get_type(name);
        img->raw = tensor_new(sys_strdup(name), TENSOR_TYPE_UINT8);
        int height, width, channels;
        uint8_t * data = stbi_load(name, &height, &width, &channels, 0);
        tensor_reshape(img->raw, 4, (int[]){1, channels, height, width});
        tensor_apply(img->raw, (void*)data, channels * height * width);
        free(data);
        data = NULL;
        return img;
    }
    return NULL;
}

void image_save(image_t* img, const char* name) {
    if(img && name && img->raw) {
        switch(image_get_type(name)) {
            case IMAGE_TYPE_BMP:
                stbi_write_bmp(name, img->raw->dims[2], img->raw->dims[3], img->raw->dims[1], img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_JPG:
                stbi_write_jpg(name, img->raw->dims[2], img->raw->dims[3], img->raw->dims[1], img->raw->datas, 100);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_PNG:
                stbi_write_png(name, img->raw->dims[2], img->raw->dims[3], img->raw->dims[1], img->raw->datas, img->raw->dims[2] * img->raw->dims[1]);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_TGA:
                stbi_write_tga(name, img->raw->dims[2], img->raw->dims[3], img->raw->dims[1], img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_HDR:
                stbi_write_hdr(name, img->raw->dims[2], img->raw->dims[3], img->raw->dims[1], img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            default: break;
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