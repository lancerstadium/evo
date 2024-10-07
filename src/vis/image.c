#include <string.h>
#include <stdio.h>
#include <evo.h>
#include <evo/util/log.h>
#include <evo/util/sys.h>
#include <evo/util/math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
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
                        LOG_INFO("%3d ", data[i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] + r * img->raw->dims[2] * img->raw->dims[3] + c * img->raw->dims[3] + s]);
                    }
                    LOG_INFO("\n");
                }
                LOG_INFO("\n");
            }
        } else {
            for (int s = 0; s < img->raw->dims[1]; ++s) {
                for (int r = 0; r < img->raw->dims[2]; ++r) {
                    for (int c = 0; c < img->raw->dims[3]; ++c) {
                        LOG_INFO("%3d ", data[i * img->raw->dims[1] * img->raw->dims[2] * img->raw->dims[3] + s * img->raw->dims[2] * img->raw->dims[3] + r * img->raw->dims[3] + c]);
                    }
                    LOG_INFO("\n");
                }
                LOG_INFO("\n");
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
    if(!ext) return IMAGE_TYPE_UNKNOWN;
    if(strcmp(ext, "bmp") == 0) {
        return IMAGE_TYPE_BMP;
    } else if(strcmp(ext, "jpg") == 0 || strcmp(ext, "JPG") == 0 || strcmp(ext, "jepg") == 0 || strcmp(ext, "JEPG") == 0) {
        return IMAGE_TYPE_JPG;
    } else if(strcmp(ext, "png") == 0 || strcmp(ext, "PNG") == 0) {
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
                        float temp = data[(is_valid ? frame : i) * ts->dims[2] * ts->dims[3] + x * ts->dims[3] + y];
                        datas[i * dims[1] * dims[2] + x * dims[2] + y] = color_mix_heat(temp);
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
        if(!data) {
            LOG_ERR("image data load fail: %s\n", name);
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

int image_width(image_t* img) {
    if(!img && !img->raw) return 0;
    if(img->raw->ndim == 3) {
        return img->raw->dims[2];
    } else if(img->raw->ndim == 4) {
        return img->raw->layout == 0 ? img->raw->dims[3] : img->raw->dims[2];
    }
    return 0;
}

int image_height(image_t* img) {
    if(!img && !img->raw) return 0;
    if(img->raw->ndim == 3) {
        return img->raw->dims[1];
    } else if(img->raw->ndim == 4) {
        return img->raw->layout == 0 ? img->raw->dims[2] : img->raw->dims[1];
    }
    return 0;
}

void image_save_channel(image_t* img, const char* name, int channel) {
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
        tensor_t *new_ts = tensor_new(img->raw->name, img->raw->type);
        if(img->raw->layout == 0) {
            new_ts = tensor_nchw2nhwc(img->raw);
        } else {
            tensor_reshape(new_ts, img->raw->ndim, img->raw->dims);
            tensor_copy(new_ts, img->raw);
        }
        switch(image_get_type(name)) {
            case IMAGE_TYPE_BMP:
                stbi_write_bmp(name, new_ts->dims[2], new_ts->dims[1], new_ts->dims[3], new_ts->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_JPG:
                stbi_write_jpg(name, new_ts->dims[2], new_ts->dims[1], new_ts->dims[3], new_ts->datas, 100);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_PNG:
                stbi_write_png(name, new_ts->dims[2], new_ts->dims[1], new_ts->dims[3], new_ts->datas, new_ts->dims[2] * new_ts->dims[3]);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_TGA:
                stbi_write_tga(name, new_ts->dims[2], new_ts->dims[1], new_ts->dims[3], new_ts->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_HDR:
                stbi_write_hdr(name, new_ts->dims[2], new_ts->dims[1], new_ts->dims[3], new_ts->datas);
                LOG_INFO("Image save: %s\n", name);
                break;
            case IMAGE_TYPE_GIF:
                attribute_t* deloys_attr = image_get_attr(img, "deloys");
                if(deloys_attr) {
                    int width = new_ts->dims[2];
                    int height = new_ts->dims[1];
                    int channel = new_ts->dims[3];
                    int64_t* deloys = deloys_attr->is;
                    uint8_t* datas = new_ts->datas;
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
        tensor_free(new_ts);
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

void image_crop_center(image_t* img, int crop_width, int crop_height) {
    if(!img || !img->raw || crop_height <= 0 || crop_width <= 0) return;
    int width, height, channel;
    if(img->raw->layout == 1) {
        width = img->raw->dims[2];
        height = img->raw->dims[1];
        channel = img->raw->dims[3];
    } else {
        width = img->raw->dims[3];
        height = img->raw->dims[2];
        channel = img->raw->dims[1];
    }
    crop_height = (crop_height <= height) ? crop_height : height;
    crop_width = (crop_width <= width) ? crop_width : width;
    unsigned char* data = sys_malloc(crop_height * crop_width * channel);
    int w_start = (width - crop_width) / 2;
    int h_start = (height - crop_height) / 2;
    int x_start = w_start + h_start * width;
    uint8_t* datas = img->raw->datas;
    for(int i = 0; i < crop_height; i++) {
        int x = x_start + width * i;
        memcpy(data + i * crop_width * channel, datas + x * channel, crop_width * channel);
    }
    // update dims
    if(img->raw->layout == 1) {
        img->raw->dims[2] = crop_width;
        img->raw->dims[1] = crop_height;
        img->raw->dims[3] = channel;
    } else {
        img->raw->dims[3] = crop_width;
        img->raw->dims[2] = crop_height;
        img->raw->dims[1] = channel;
    }
    // update strides
    for(int i = img->raw->ndim - 2; i >= 0; i--) {
        img->raw->strides[i] = img->raw->strides[i+1] * img->raw->dims[i+1];
    }
    // update datas
    img->raw->ndata = crop_height * crop_width * channel;
    free(img->raw->datas);
    img->raw->datas = data;
}

image_t* image_copy(image_t* img) {
    if(!img || !img->raw) return NULL;
    tensor_t* raw = tensor_new(img->raw->name, img->raw->type);
    tensor_copy(raw, img->raw);
    image_t* new_img = image_from_tensor(raw);
    return new_img;
}

void image_resize(image_t* img, int resize_width, int resize_height) {
    if(!img || !img->raw || resize_height <= 0 || resize_width <= 0) return;
    int width, height, channel;
    if(img->raw->layout == 1) {
        width = img->raw->dims[2];
        height = img->raw->dims[1];
        channel = img->raw->dims[3];
    } else {
        width = img->raw->dims[3];
        height = img->raw->dims[2];
        channel = img->raw->dims[1];
    }
    unsigned char* data = sys_malloc(resize_height * resize_width * channel);
    stbir_resize_uint8_linear(img->raw->datas, width, height, 0, data, resize_width, resize_height, 0, channel);
    // update dims
    if(img->raw->layout == 1) {
        img->raw->dims[2] = resize_width;
        img->raw->dims[1] = resize_height;
        img->raw->dims[3] = channel;
    } else {
        img->raw->dims[3] = resize_width;
        img->raw->dims[2] = resize_height;
        img->raw->dims[1] = channel;
    }
    // update strides
    for(int i = img->raw->ndim - 2; i >= 0; i--) {
        img->raw->strides[i] = img->raw->strides[i+1] * img->raw->dims[i+1];
    }
    // update datas
    img->raw->ndata = resize_height * resize_width * channel;
    free(img->raw->datas);
    img->raw->datas = data;
}

image_t* image_merge(image_t* a, image_t* b, float alpha) {
    if(!a || !b || !a->raw || !b->raw) return NULL;
    if(a->raw->layout != 1 || b->raw->layout != 1 || a->raw->dims[1] != b->raw->dims[1] || a->raw->dims[2] != b->raw->dims[2]) return a;
    uint8_t* data_a = a->raw->datas;
    uint8_t* data_b = b->raw->datas;
    if(alpha > 1.0f) alpha = 1.0f;
    if(alpha < 0.0f) alpha = 0.0f;
    if(a->raw->dims[3] != b->raw->dims[3]) {
        int channel_a = a->raw->dims[3];
        int channel_b = b->raw->dims[3];
        int ndata_a = a->raw->dims[1] * a->raw->dims[2] * channel_b;
        uint8_t* datas_a = sys_malloc(ndata_a);
        memset(datas_a, 0, ndata_a);
        a->raw->dims[3] = channel_b;                            // update dims
        for(int i = a->raw->ndim - 2; i >= 0; i--) {            // update strides
            a->raw->strides[i] = a->raw->strides[i+1] * a->raw->dims[i+1];
        }
        for(int i = 0; i < a->raw->ndata / channel_a; i++) {    // update datas
            float ratio = (float)data_b[i * channel_b + channel_b - 1] / 255;
            float sum = 1 - alpha;
            float ratio2 = (1 - ratio) / sum;
            for(int c = 0; c < channel_a; c++) {
                uint8_t a_part = data_a[i * channel_a + c];
                uint8_t b_part = data_b[i * channel_b + c];
                if(ratio < alpha) {
                    datas_a[i * channel_b + c] = a_part;
                } else {
                    datas_a[i * channel_b + c] = a_part * ratio2 + b_part * (1 - ratio2);
                }
            }
            datas_a[i * channel_b + channel_b - 1] = ratio < alpha ? 0xff : 0xff * ratio2 + data_b[i * channel_b + channel_b - 1] * (1 - ratio2);
        }
        a->raw->ndata = ndata_a;
        free(a->raw->datas);
        a->raw->datas = (void*)datas_a;
    } else {
        for(int i = 0; i < MIN(a->raw->ndata, b->raw->ndata); i++) {
            data_a[i] = data_a[i] * (1.0f - alpha) + data_b[i] * alpha;
        }
    }
    return a;
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

void image_to_grey(image_t* img) {
    if(!img || !img->raw || img->raw->layout != 1 || img->raw->type != TENSOR_TYPE_UINT8 || img->raw->ndim < 4 || img->raw->dims[3] != 3) return;

    int width = image_width(img);
    int height = image_height(img);
    int channel = img->raw->dims[3];
    uint8_t *grey_img = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    if (!grey_img) return;
    uint8_t* org_img = img->raw->datas;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channel;
            int r = org_img[idx];
            int g = org_img[idx + 1];
            int b = org_img[idx + 2];
            int grey = (r * 77 + g * 150 + b * 29) / 256; // 转换为灰度值
            grey_img[y * width + x] = grey;
        }
    }
    
    tensor_reshape(img->raw, 4, (int[]){1, height, width, 1});
    uint8_t* new_img = img->raw->datas;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            new_img[idx] = grey_img[idx];
        }
    }

    // 释放原始图像数据
    free(grey_img);
}

void image_halftone_ordered_dithering(image_t* img) {
    if (!img || !img->raw || img->raw->type != TENSOR_TYPE_UINT8 || img->raw->ndim < 4 || img->raw->dims[3] != 1) return;

    int width = image_width(img);
    int height = image_height(img);
    uint8_t* org_img = img->raw->datas;

    // 定义一个4x4的Bayer矩阵
    int bayer_matrix[4][4] = {
        {  0,  8,  2, 10 },
        { 12,  4, 14,  6 },
        {  3, 11,  1,  9 },
        { 15,  7, 13,  5 }
    };

    // 归一化Bayer矩阵的值
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int threshold = bayer_matrix[y % 4][x % 4] * 16;
            int original = org_img[idx];
            int quantized = original > threshold ? 255 : 0;
            org_img[idx] = quantized;
        }
    }
}

void image_halftone_floyd_steinberg(image_t* img) {
    if(!img || !img->raw || img->raw->type != TENSOR_TYPE_UINT8 || img->raw->ndim < 4 || img->raw->dims[3] != 1) return;
    int width = image_width(img);
    int height = image_height(img);
    uint8_t* org_img = img->raw->datas;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int original = org_img[idx];
            int quantized = original > 127 ? 255 : 0; // 阈值化处理
            org_img[idx] = quantized;
            // Floyd–Steinberg dithering error distribution
            int error = original - quantized;

            if (x < width - 1) {
                org_img[idx + 1] += error * 7 / 16;
            }
            if (x > 0 && y < height - 1) {
                org_img[idx - 1 + width] += error * 3 / 16;
            }
            if (y < height - 1) {
                org_img[idx + width] += error * 5 / 16;
            }
            if (x < width - 1 && y < height - 1) {
                org_img[idx + 1 + width] += error * 1 / 16;
            }
        }
    }
}


void image_halftone_ostromoukhov(image_t* img) {
    if (!img || !img->raw || img->raw->type != TENSOR_TYPE_UINT8 || img->raw->ndim < 4 || img->raw->dims[3] != 1) return;

    int width = image_width(img);
    int height = image_height(img);
    uint8_t* img_data = img->raw->datas;

    // 定义存储误差的缓冲区
    float* carry_line_0 = (float*)calloc(width, sizeof(float));
    float* carry_line_1 = (float*)calloc(width, sizeof(float));

    // 从文件提取的系数表
    int var_coefs_tab[256][4] = {
       { 13,     0,     5,    18,},     /*    0 */
       { 13,     0,     5,    18,},     /*    1 */
       { 21,     0,    10,    31,},     /*    2 */
       {  7,     0,     4,    11,},     /*    3 */
       {  8,     0,     5,    13,},     /*    4 */
       { 47,     3,    28,    78,},     /*    5 */
       { 23,     3,    13,    39,},     /*    6 */
       { 15,     3,     8,    26,},     /*    7 */
       { 22,     6,    11,    39,},     /*    8 */
       { 43,    15,    20,    78,},     /*    9 */
       {  7,     3,     3,    13,},     /*   10 */
       {501,   224,   211,   936,},     /*   11 */
       {249,   116,   103,   468,},     /*   12 */
       {165,    80,    67,   312,},     /*   13 */
       {123,    62,    49,   234,},     /*   14 */
       {489,   256,   191,   936,},     /*   15 */
       { 81,    44,    31,   156,},     /*   16 */
       {483,   272,   181,   936,},     /*   17 */
       { 60,    35,    22,   117,},     /*   18 */
       { 53,    32,    19,   104,},     /*   19 */
       {237,   148,    83,   468,},     /*   20 */
       {471,   304,   161,   936,},     /*   21 */
       {  3,     2,     1,     6,},     /*   22 */
       {459,   304,   161,   924,},     /*   23 */
       { 38,    25,    14,    77,},     /*   24 */
       {453,   296,   175,   924,},     /*   25 */
       {225,   146,    91,   462,},     /*   26 */
       {149,    96,    63,   308,},     /*   27 */
       {111,    71,    49,   231,},     /*   28 */
       { 63,    40,    29,   132,},     /*   29 */
       { 73,    46,    35,   154,},     /*   30 */
       {435,   272,   217,   924,},     /*   31 */
       {108,    67,    56,   231,},     /*   32 */
       { 13,     8,     7,    28,},     /*   33 */
       {213,   130,   119,   462,},     /*   34 */
       {423,   256,   245,   924,},     /*   35 */
       {  5,     3,     3,    11,},     /*   36 */
       {281,   173,   162,   616,},     /*   37 */
       {141,    89,    78,   308,},     /*   38 */
       {283,   183,   150,   616,},     /*   39 */
       { 71,    47,    36,   154,},     /*   40 */
       {285,   193,   138,   616,},     /*   41 */
       { 13,     9,     6,    28,},     /*   42 */
       { 41,    29,    18,    88,},     /*   43 */
       { 36,    26,    15,    77,},     /*   44 */
       {289,   213,   114,   616,},     /*   45 */
       {145,   109,    54,   308,},     /*   46 */
       {291,   223,   102,   616,},     /*   47 */
       { 73,    57,    24,   154,},     /*   48 */
       {293,   233,    90,   616,},     /*   49 */
       { 21,    17,     6,    44,},     /*   50 */
       {295,   243,    78,   616,},     /*   51 */
       { 37,    31,     9,    77,},     /*   52 */
       { 27,    23,     6,    56,},     /*   53 */
       {149,   129,    30,   308,},     /*   54 */
       {299,   263,    54,   616,},     /*   55 */
       { 75,    67,    12,   154,},     /*   56 */
       { 43,    39,     6,    88,},     /*   57 */
       {151,   139,    18,   308,},     /*   58 */
       {303,   283,    30,   616,},     /*   59 */
       { 38,    36,     3,    77,},     /*   60 */
       {305,   293,    18,   616,},     /*   61 */
       {153,   149,     6,   308,},     /*   62 */
       {307,   303,     6,   616,},     /*   63 */
       {  1,     1,     0,     2,},     /*   64 */
       {101,   105,     2,   208,},     /*   65 */
       { 49,    53,     2,   104,},     /*   66 */
       { 95,   107,     6,   208,},     /*   67 */
       { 23,    27,     2,    52,},     /*   68 */
       { 89,   109,    10,   208,},     /*   69 */
       { 43,    55,     6,   104,},     /*   70 */
       { 83,   111,    14,   208,},     /*   71 */
       {  5,     7,     1,    13,},     /*   72 */
       {172,   181,    37,   390,},     /*   73 */
       { 97,    76,    22,   195,},     /*   74 */
       { 72,    41,    17,   130,},     /*   75 */
       {119,    47,    29,   195,},     /*   76 */
       {  4,     1,     1,     6,},     /*   77 */
       {  4,     1,     1,     6,},     /*   78 */
       {  4,     1,     1,     6,},     /*   79 */
       {  4,     1,     1,     6,},     /*   80 */
       {  4,     1,     1,     6,},     /*   81 */
       {  4,     1,     1,     6,},     /*   82 */
       {  4,     1,     1,     6,},     /*   83 */
       {  4,     1,     1,     6,},     /*   84 */
       {  4,     1,     1,     6,},     /*   85 */
       { 65,    18,    17,   100,},     /*   86 */
       { 95,    29,    26,   150,},     /*   87 */
       {185,    62,    53,   300,},     /*   88 */
       { 30,    11,     9,    50,},     /*   89 */
       { 35,    14,    11,    60,},     /*   90 */
       { 85,    37,    28,   150,},     /*   91 */
       { 55,    26,    19,   100,},     /*   92 */
       { 80,    41,    29,   150,},     /*   93 */
       {155,    86,    59,   300,},     /*   94 */
       {  5,     3,     2,    10,},     /*   95 */
       {  5,     3,     2,    10,},     /*   96 */
       {  5,     3,     2,    10,},     /*   97 */
       {  5,     3,     2,    10,},     /*   98 */
       {  5,     3,     2,    10,},     /*   99 */
       {  5,     3,     2,    10,},     /*  100 */
       {  5,     3,     2,    10,},     /*  101 */
       {  5,     3,     2,    10,},     /*  102 */
       {  5,     3,     2,    10,},     /*  103 */
       {  5,     3,     2,    10,},     /*  104 */
       {  5,     3,     2,    10,},     /*  105 */
       {  5,     3,     2,    10,},     /*  106 */
       {  5,     3,     2,    10,},     /*  107 */
       {305,   176,   119,   600,},     /*  108 */
       {155,    86,    59,   300,},     /*  109 */
       {105,    56,    39,   200,},     /*  110 */
       { 80,    41,    29,   150,},     /*  111 */
       { 65,    32,    23,   120,},     /*  112 */
       { 55,    26,    19,   100,},     /*  113 */
       {335,   152,   113,   600,},     /*  114 */
       { 85,    37,    28,   150,},     /*  115 */
       {115,    48,    37,   200,},     /*  116 */
       { 35,    14,    11,    60,},     /*  117 */
       {355,   136,   109,   600,},     /*  118 */
       { 30,    11,     9,    50,},     /*  119 */
       {365,   128,   107,   600,},     /*  120 */
       {185,    62,    53,   300,},     /*  121 */
       { 25,     8,     7,    40,},     /*  122 */
       { 95,    29,    26,   150,},     /*  123 */
       {385,   112,   103,   600,},     /*  124 */
       { 65,    18,    17,   100,},     /*  125 */
       {395,   104,   101,   600,},     /*  126 */
       {  4,     1,     1,     6,},     /*  127 */
       {  4,     1,     1,     6,},     /*  128 */
       {395,   104,   101,   600,},     /*  129 */
       { 65,    18,    17,   100,},     /*  130 */
       {385,   112,   103,   600,},     /*  131 */
       { 95,    29,    26,   150,},     /*  132 */
       { 25,     8,     7,    40,},     /*  133 */
       {185,    62,    53,   300,},     /*  134 */
       {365,   128,   107,   600,},     /*  135 */
       { 30,    11,     9,    50,},     /*  136 */
       {355,   136,   109,   600,},     /*  137 */
       { 35,    14,    11,    60,},     /*  138 */
       {115,    48,    37,   200,},     /*  139 */
       { 85,    37,    28,   150,},     /*  140 */
       {335,   152,   113,   600,},     /*  141 */
       { 55,    26,    19,   100,},     /*  142 */
       { 65,    32,    23,   120,},     /*  143 */
       { 80,    41,    29,   150,},     /*  144 */
       {105,    56,    39,   200,},     /*  145 */
       {155,    86,    59,   300,},     /*  146 */
       {305,   176,   119,   600,},     /*  147 */
       {  5,     3,     2,    10,},     /*  148 */
       {  5,     3,     2,    10,},     /*  149 */
       {  5,     3,     2,    10,},     /*  150 */
       {  5,     3,     2,    10,},     /*  151 */
       {  5,     3,     2,    10,},     /*  152 */
       {  5,     3,     2,    10,},     /*  153 */
       {  5,     3,     2,    10,},     /*  154 */
       {  5,     3,     2,    10,},     /*  155 */
       {  5,     3,     2,    10,},     /*  156 */
       {  5,     3,     2,    10,},     /*  157 */
       {  5,     3,     2,    10,},     /*  158 */
       {  5,     3,     2,    10,},     /*  159 */
       {  5,     3,     2,    10,},     /*  160 */
       {155,    86,    59,   300,},     /*  161 */
       { 80,    41,    29,   150,},     /*  162 */
       { 55,    26,    19,   100,},     /*  163 */
       { 85,    37,    28,   150,},     /*  164 */
       { 35,    14,    11,    60,},     /*  165 */
       { 30,    11,     9,    50,},     /*  166 */
       {185,    62,    53,   300,},     /*  167 */
       { 95,    29,    26,   150,},     /*  168 */
       { 65,    18,    17,   100,},     /*  169 */
       {  4,     1,     1,     6,},     /*  170 */
       {  4,     1,     1,     6,},     /*  171 */
       {  4,     1,     1,     6,},     /*  172 */
       {  4,     1,     1,     6,},     /*  173 */
       {  4,     1,     1,     6,},     /*  174 */
       {  4,     1,     1,     6,},     /*  175 */
       {  4,     1,     1,     6,},     /*  176 */
       {  4,     1,     1,     6,},     /*  177 */
       {  4,     1,     1,     6,},     /*  178 */
       {119,    47,    29,   195,},     /*  179 */
       { 72,    41,    17,   130,},     /*  180 */
       { 97,    76,    22,   195,},     /*  181 */
       {172,   181,    37,   390,},     /*  182 */
       {  5,     7,     1,    13,},     /*  183 */
       { 83,   111,    14,   208,},     /*  184 */
       { 43,    55,     6,   104,},     /*  185 */
       { 89,   109,    10,   208,},     /*  186 */
       { 23,    27,     2,    52,},     /*  187 */
       { 95,   107,     6,   208,},     /*  188 */
       { 49,    53,     2,   104,},     /*  189 */
       {101,   105,     2,   208,},     /*  190 */
       {  1,     1,     0,     2,},     /*  191 */
       {307,   303,     6,   616,},     /*  192 */
       {153,   149,     6,   308,},     /*  193 */
       {305,   293,    18,   616,},     /*  194 */
       { 38,    36,     3,    77,},     /*  195 */
       {303,   283,    30,   616,},     /*  196 */
       {151,   139,    18,   308,},     /*  197 */
       { 43,    39,     6,    88,},     /*  198 */
       { 75,    67,    12,   154,},     /*  199 */
       {299,   263,    54,   616,},     /*  200 */
       {149,   129,    30,   308,},     /*  201 */
       { 27,    23,     6,    56,},     /*  202 */
       { 37,    31,     9,    77,},     /*  203 */
       {295,   243,    78,   616,},     /*  204 */
       { 21,    17,     6,    44,},     /*  205 */
       {293,   233,    90,   616,},     /*  206 */
       { 73,    57,    24,   154,},     /*  207 */
       {291,   223,   102,   616,},     /*  208 */
       {145,   109,    54,   308,},     /*  209 */
       {289,   213,   114,   616,},     /*  210 */
       { 36,    26,    15,    77,},     /*  211 */
       { 41,    29,    18,    88,},     /*  212 */
       { 13,     9,     6,    28,},     /*  213 */
       {285,   193,   138,   616,},     /*  214 */
       { 71,    47,    36,   154,},     /*  215 */
       {283,   183,   150,   616,},     /*  216 */
       {141,    89,    78,   308,},     /*  217 */
       {281,   173,   162,   616,},     /*  218 */
       {  5,     3,     3,    11,},     /*  219 */
       {423,   256,   245,   924,},     /*  220 */
       {213,   130,   119,   462,},     /*  221 */
       { 13,     8,     7,    28,},     /*  222 */
       {108,    67,    56,   231,},     /*  223 */
       {435,   272,   217,   924,},     /*  224 */
       { 73,    46,    35,   154,},     /*  225 */
       { 63,    40,    29,   132,},     /*  226 */
       {111,    71,    49,   231,},     /*  227 */
       {149,    96,    63,   308,},     /*  228 */
       {225,   146,    91,   462,},     /*  229 */
       {453,   296,   175,   924,},     /*  230 */
       { 38,    25,    14,    77,},     /*  231 */
       {459,   304,   161,   924,},     /*  232 */
       {  3,     2,     1,     6,},     /*  233 */
       {471,   304,   161,   936,},     /*  234 */
       {237,   148,    83,   468,},     /*  235 */
       { 53,    32,    19,   104,},     /*  236 */
       { 60,    35,    22,   117,},     /*  237 */
       {483,   272,   181,   936,},     /*  238 */
       { 81,    44,    31,   156,},     /*  239 */
       {489,   256,   191,   936,},     /*  240 */
       {123,    62,    49,   234,},     /*  241 */
       {165,    80,    67,   312,},     /*  242 */
       {249,   116,   103,   468,},     /*  243 */
       {501,   224,   211,   936,},     /*  244 */
       {  7,     3,     3,    13,},     /*  245 */
       { 43,    15,    20,    78,},     /*  246 */
       { 22,     6,    11,    39,},     /*  247 */
       { 15,     3,     8,    26,},     /*  248 */
       { 23,     3,    13,    39,},     /*  249 */
       { 47,     3,    28,    78,},     /*  250 */
       {  8,     0,     5,    13,},     /*  251 */
       {  7,     0,     4,    11,},     /*  252 */
       { 21,     0,    10,    31,},     /*  253 */
       { 13,     0,     5,    18,},     /*  254 */
       { 13,     0,     5,    18 },     /*  255 */
    };

    // 遍历图像的每个像素
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            // 获取像素强度并根据它访问 var_coefs_tab
            int intensity = img_data[idx];

            // 使用指针直接访问系数
            int* coefs = var_coefs_tab[intensity];

            // 当前像素值（缩放至 [0, 1] 以计算误差）
            float old_pixel = img_data[idx] / 255.0f;

            // 加上前面扩散过来的误差
            float new_pixel = old_pixel + carry_line_0[x];

            // 确定四舍五入到黑或白
            uint8_t halftone_pixel = (new_pixel > 0.5f) ? 255 : 0;
            img_data[idx] = halftone_pixel;

            // 计算量化误差
            float error = new_pixel - (halftone_pixel / 255.0f);

            // 使用指针系数分配误差
            if (x + 1 < width) {
                carry_line_0[x + 1] += coefs[0] / (float)coefs[3] * error; // 右邻居
            }
            if (y + 1 < height) {
                carry_line_1[x] += coefs[2] / (float)coefs[3] * error;     // 下邻居
                if (x > 0) {
                    carry_line_1[x - 1] += coefs[1] / (float)coefs[3] * error; // 左下
                }
                if (x + 1 < width) {
                    carry_line_1[x + 1] += 1.0f / (float)coefs[3] * error;   // 右下
                }
            }
        }

        // 交换误差缓冲区
        float* temp = carry_line_0;
        carry_line_0 = carry_line_1;
        carry_line_1 = temp;

        // 清除下一行的误差缓冲区
        memset(carry_line_1, 0, width * sizeof(float));
    }

    // 释放误差缓冲区
    free(carry_line_0);
    free(carry_line_1);
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