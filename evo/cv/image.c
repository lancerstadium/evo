#include <string.h>
#include <stdio.h>
#include "../evo.h"
#include "../util/log.h"
#include "../util/sys.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "gifenc.h"

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
        if(img->raw->layout == 1) {
            for (int s = 0; s < img->raw->dims[3]; ++s) {
                for (int r = 0; r < img->raw->dims[1]; ++r) {
                    for (int c = 0; c < img->raw->dims[2]; ++c) {
                        printf("%3d ", data[i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] + r * img->raw->dims[2] * img->raw->dims[3] + c * img->raw->dims[3] + s]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        } else {
            for (int s = 0; s < img->raw->dims[1]; ++s) {
                for (int r = 0; r < img->raw->dims[2]; ++r) {
                    for (int c = 0; c < img->raw->dims[3]; ++c) {
                        printf("%3d ", data[i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] + s * img->raw->dims[2] * img->raw->dims[3] + r * img->raw->dims[3] + c]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
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
    } else if(strcmp(ext, "gif") == 0) {
        return IMAGE_TYPE_GIF;
    } else {
        return IMAGE_TYPE_UNKNOWN;
    }
}

image_t* image_from_tensor(tensor_t *ts) {
    image_t* img = (image_t*)malloc(sizeof(image_t));
    if(img && ts) {
        tensor_t *new_ts = tensor_new(sys_strdup(ts->name), TENSOR_TYPE_UINT8);
        int ndim = 4;
        int dims[4];
        dims[0] = 1;
        if(ts->ndim <= 2) {
            return NULL;
        }
        if(ts->layout != 1) {
            ts = tensor_nchw2nhwc(ts);
        }
        if(ts->ndim == 3) {
            dims[1] = ts->dims[0];
            dims[2] = ts->dims[1];
            dims[3] = ts->dims[2];
        }
        dims[1] = ts->dims[1];
        dims[2] = ts->dims[2];
        dims[3] = ts->dims[3];
        tensor_reshape(new_ts, ndim, dims);
        new_ts->ndata = ts->dims[1] * ts->dims[2] * ts->dims[3];
        new_ts->datas = malloc(new_ts->ndata);
        new_ts->layout = 1;
        if(ts->type == TENSOR_TYPE_FLOAT32) {
            float* data = ts->datas;
            uint8_t* datas = new_ts->datas;
            for(int i = 0; i < new_ts->ndata; i++) {
                datas[i] = (uint8_t)data[i];
            }
        } else if(ts->type == TENSOR_TYPE_UINT8) {
            uint8_t* data = ts->datas;
            uint8_t* datas = new_ts->datas;
            for(int i = 0; i < new_ts->ndata; i++) {
                datas[i] = (uint8_t)data[i];
            }
        }
        img->name = sys_strdup(ts->name);
        img->attr_vec = vector_create();
        img->type = IMAGE_TYPE_UNKNOWN;
        img->raw = new_ts;
        if(ts->layout != 1) {
            tensor_free(ts);
        }
        return img;
    }
    return NULL;
}

image_t* image_heatmap(tensor_t *ts, int frame) {
    image_t* img = (image_t*)malloc(sizeof(image_t));
    if(img && ts) {
        tensor_t *new_ts = tensor_new(sys_strdup(ts->name), TENSOR_TYPE_UINT8);
        int ndim = 4;
        int dims[4];
        bool is_valid = frame <= ts->dims[0] - 1 && frame >= 0;
        if(is_valid) {
            dims[0] = 1;
        } else {
            dims[0] = ts->dims[0];
        }
        dims[3] = 4;
        if(ts->ndim <= 2) {
            return NULL;
        } else if(ts->ndim == 3) {
            dims[1] = ts->dims[1];
            dims[2] = ts->dims[2];
        }
        dims[1] = ts->dims[2];
        dims[2] = ts->dims[3];
        tensor_reshape(new_ts, ndim, dims);
        new_ts->ndata = dims[0] * dims[1] * dims[2] * 4;
        new_ts->datas = malloc(new_ts->ndata);
        if(ts->type == TENSOR_TYPE_FLOAT32) {
            float* data = ts->datas;
            uint32_t* datas = new_ts->datas;
            for(int i = 0; i < dims[0]; i++) {
                for(int x = 0; x < dims[1]; x++) {
                    for(int y = 0; y < dims[2]; y++) {
                        datas[i * dims[1] * dims[2] + x * dims[2] + y] = color_mix(0x00ff0000, 0xf00000ff, data[(is_valid ? frame : i) * ts->dims[2] * ts->dims[3] + x * ts->dims[3] + y]);
                    }
                }
            }
        }
        img->name = sys_strdup(ts->name);
        img->attr_vec = vector_create();
        img->type = IMAGE_TYPE_UNKNOWN;
        img->raw = new_ts;
        img->raw->layout = 1;
        return img;
    }
    return NULL;
}

image_t* image_blank(const char* name, size_t width, size_t height) {
    image_t* img = (image_t*)malloc(sizeof(image_t));
    if(img) {
        img->name = sys_strdup(name);
        img->attr_vec = vector_create();
        img->type = IMAGE_TYPE_UNKNOWN;
        img->raw = tensor_new(sys_strdup(name), TENSOR_TYPE_UINT8);
        tensor_reshape(img->raw, 4, (int[]){1, height, width, 4});
        uint8_t * data = sys_malloc(4 * height * width * sizeof(uint8_t));
        tensor_apply(img->raw, (void*)data, 4 * height * width);
        img->raw->layout = 1;
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
        int frame = 1;
        uint8_t * data = NULL;
        if(img->type == IMAGE_TYPE_GIF) {
            int *delays = NULL;
            int64_t *new_delays = NULL;
            data = stbi_load_gif(name, &delays, &width, &height, &frame, &channels, 0);
            if(delays) {
                new_delays = malloc(frame * sizeof(int64_t));
                for(int i = 0; i < frame; i++) {
                    new_delays[i] = (int64_t)delays[i];
                }
                free(delays);
            }
            attribute_t* delays_attr = attribute_ints("delays", new_delays, frame);
            vector_add(&img->attr_vec, delays_attr);
        } else {
            data = stbi_load(name, &width, &height, &channels, 0);
        }
        tensor_reshape(img->raw, 4, (int[]){frame, height, width, channels});
        tensor_apply(img->raw, (void*)data, frame * channels * height * width);
        img->raw->layout = 1;
        free(data);
        data = NULL;
        return img;
    }
    return NULL;
}

image_t* image_extract_channel(image_t* img, int channel) {
    image_t* new_img = (image_t*)malloc(sizeof(image_t));
    if(new_img && img) {
        if(channel >= img->raw->dims[3]) channel = 0;
        new_img->name = sys_strdup(img->name);
        new_img->attr_vec = vector_create();
        new_img->type = img->type;
        new_img->raw = tensor_new(sys_strdup(img->name), TENSOR_TYPE_UINT8);
        tensor_reshape(new_img->raw, 4, (int[]){1, img->raw->dims[1], img->raw->dims[2], 1});
        new_img->raw->ndata = img->raw->dims[1] * img->raw->dims[2];
        new_img->raw->datas = malloc(new_img->raw->ndata);
        new_img->raw->layout = img->raw->layout;
        uint8_t* data = new_img->raw->datas;
        uint8_t* datas = img->raw->datas;
        for(int i = 0; i < new_img->raw->ndata; i++) {
            data[i] = datas[i * img->raw->dims[3] + channel];
        }
        return new_img;
    }
    return NULL;
}

image_t* image_channel(image_t* img, int channel) {
    image_t* new_img = (image_t*)malloc(sizeof(image_t));
    if(new_img && img && img->raw && img->raw->layout == 1) {
        if(channel >= img->raw->dims[3]) channel = 0;
        new_img->name = sys_strdup(img->name);
        new_img->attr_vec = vector_create();
        new_img->type = img->type;
        new_img->raw = tensor_new(sys_strdup(img->name), TENSOR_TYPE_UINT8);
        tensor_reshape(new_img->raw, 4, (int[]){1, img->raw->dims[1], img->raw->dims[2], img->raw->dims[3]});
        new_img->raw->ndata = img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3];
        new_img->raw->datas = malloc(new_img->raw->ndata);
        new_img->raw->layout = img->raw->layout;
        memset(new_img->raw->datas, 0, new_img->raw->ndata);
        uint8_t* data = new_img->raw->datas;
        uint8_t* datas = img->raw->datas;
        for(int i = 0; i < new_img->raw->ndata / img->raw->dims[3]; i++) {
            data[i * img->raw->dims[3] + channel] = datas[i * img->raw->dims[3] + channel];
        }
        return new_img;
    }
    return NULL;
}

void image_save_grey(image_t* img, const char* name, int channel) {
    if(img && name && img->raw) {
        if(img->raw->type != TENSOR_TYPE_UINT8) {
            LOG_WARN("Image save warn: only support uint8 tensor!\n");
            return;
        }
        image_t* grey_img = image_extract_channel(img, channel);
        switch(image_get_type(name)) {
            case IMAGE_TYPE_BMP:
                stbi_write_bmp(name, img->raw->dims[2], img->raw->dims[1], 1, grey_img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_JPG:
                stbi_write_jpg(name, img->raw->dims[2], img->raw->dims[1], 1, grey_img->raw->datas, 100);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_PNG:
                stbi_write_png(name, img->raw->dims[2], img->raw->dims[1], 1, grey_img->raw->datas, img->raw->dims[2] * 1);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_TGA:
                stbi_write_tga(name, img->raw->dims[2], img->raw->dims[1], 1, grey_img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_HDR:
                stbi_write_hdr(name, img->raw->dims[2], img->raw->dims[1], 1, grey_img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            default: break;
        }
        image_free(grey_img);
    }
}

void image_save(image_t* img, const char* name) {
    if(img && name && img->raw) {
        if(img->raw->type != TENSOR_TYPE_UINT8) {
            LOG_WARN("Image save warn: only support uint8 tensor!\n");
            return;
        }
        if(img->raw->layout == 0) {
            tensor_t *new_ts = tensor_nchw2nhwc(img->raw);
            tensor_free(img->raw);
            img->raw = new_ts;
        }
        switch(image_get_type(name)) {
            case IMAGE_TYPE_BMP:
                stbi_write_bmp(name, img->raw->dims[2], img->raw->dims[1], img->raw->dims[3], img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_JPG:
                stbi_write_jpg(name, img->raw->dims[2], img->raw->dims[1], img->raw->dims[3], img->raw->datas, 100);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_PNG:
                stbi_write_png(name, img->raw->dims[2], img->raw->dims[1], img->raw->dims[3], img->raw->datas, img->raw->dims[2] * img->raw->dims[3]);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_TGA:
                stbi_write_tga(name, img->raw->dims[2], img->raw->dims[1], img->raw->dims[3], img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_HDR:
                stbi_write_hdr(name, img->raw->dims[2], img->raw->dims[1], img->raw->dims[3], img->raw->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_GIF:
                attribute_t* deloys_attr = image_get_attr(img, "deloys");
                if(deloys_attr) {
                    int width = img->raw->dims[2];
                    int height = img->raw->dims[1];
                    int channel = img->raw->dims[3];
                    int64_t* deloys = deloys_attr->is;
                    uint8_t* datas = img->raw->datas;
                    ge_GIF *gif = ge_new_gif(name, width, height, NULL, 8, -1, 0);
                    for(int i = 0; i < img->raw->dims[0]; i++) {
                        ge_render_frame(gif, datas + width * height * channel * i, channel);
                        ge_add_frame(gif, (uint16_t) deloys[i]);
                    }
                    ge_close_gif(gif);
                    LOG_INFO("Image save: %s\n", name);
                } else {
                    LOG_WARN("Image warn: gif no deloys attribute!\n");
                }
            default: break;
        }
    }
}

void image_set_deloys(image_t* img, int64_t* deloys, int len) {
    if(img && deloys) {
        attribute_t* deloys_attr = image_get_attr(img, "deloys");
        if(!deloys_attr) {
            deloys_attr = attribute_ints("deloys", deloys, len);
            vector_add(&img->attr_vec, deloys_attr);
        } else {
            if(len < deloys_attr->ni) {
                deloys_attr->ni = len;
                for(int i = 0; i < len; i++) {
                    deloys_attr->is[i] = deloys[i];
                }
            } else {
                deloys_attr->ni = len;
                // if(deloys_attr->is) free(deloys_attr->is);
                deloys_attr->is = malloc(len * sizeof(int64_t));
                for(int i = 0; i < len; i++) {
                    deloys_attr->is[i] = deloys[i];
                }
            }
        }
    }
}

void image_push(image_t* a, image_t* b) {
    if(!a || !b || !a->raw || !b->raw) return;
    if(a->raw->type != b->raw->type || !a->raw->layout || !b->raw->layout) return;
    if(a->raw->dims[1] == b->raw->dims[1] && a->raw->dims[2] == b->raw->dims[2] && a->raw->dims[3] == b->raw->dims[3]) {
        int old_dim = a->raw->dims[0];
        a->raw->dims[0] += b->raw->dims[0];
        a->raw->strides[0] += b->raw->strides[0];
        int ndata = a->raw->dims[0] * a->raw->dims[1] * a->raw->dims[2] * a->raw->dims[3];
        void *datas = malloc(ndata * tensor_type_sizeof(a->raw->type));
        if(datas) {
            memcpy(datas, a->raw->datas, old_dim * a->raw->dims[1] * a->raw->dims[2] * a->raw->dims[3]);
            memcpy(datas + old_dim * a->raw->dims[1] * a->raw->dims[2] * a->raw->dims[3], b->raw->datas, b->raw->dims[0] * a->raw->dims[1] * a->raw->dims[2] * a->raw->dims[3]);
            a->raw->ndata = ndata;
            if(a->raw->datas) free(a->raw->datas);
            a->raw->datas = datas;
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