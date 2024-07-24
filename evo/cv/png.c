#include "../evo.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


EVO_PACKED(typedef struct {
    uint8_t signature[8];   // PNG 文件签名
}) png_file_signature_t;

EVO_PACKED(typedef struct {
    uint32_t length;        // 数据块长度
    uint32_t type;          // 数据块类型
}) png_chunk_header_t;

EVO_PACKED(typedef struct {
    uint32_t width;         // 图像宽度
    uint32_t height;        // 图像高度
    uint8_t bit_depth;      // 位深度
    uint8_t color_type;     // 颜色类型
    uint8_t compression;    // 压缩方法
    uint8_t filter;         // 滤波方法
    uint8_t interlace;      // 交错方法
}) png_ihdr_chunk_t;


static inline uint32_t read_uint32(FILE* file) {
    uint32_t value;
    fread(&value, sizeof(uint32_t), 1, file);
    return __builtin_bswap32(value);
}

static inline uint8_t read_uint8(FILE* file) {
    uint8_t value;
    fread(&value, sizeof(uint8_t), 1, file);
    return value;
}

image_t* image_load_png(const char *filename) {
 FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    png_file_signature_t signature;
    if (fread(&signature, sizeof(png_file_signature_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read PNG file signature\n");
        fclose(file);
        return NULL;
    }

    // 检查 PNG 文件签名
    const uint8_t png_signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    if (memcmp(signature.signature, png_signature, sizeof(png_signature)) != 0) {
        fprintf(stderr, "Not a valid PNG file %s\n", filename);
        fclose(file);
        return NULL;
    }

    png_chunk_header_t chunk_header;
    png_ihdr_chunk_t ihdr_chunk;

    // 读取 IHDR 数据块
    if (fread(&chunk_header, sizeof(png_chunk_header_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read PNG chunk header\n");
        fclose(file);
        return NULL;
    }

    chunk_header.length = __builtin_bswap32(chunk_header.length);
    chunk_header.type = __builtin_bswap32(chunk_header.type);

    if (chunk_header.type != 0x49484452) { // IHDR
        fprintf(stderr, "First chunk is not IHDR\n");
        fclose(file);
        return NULL;
    }

    if (fread(&ihdr_chunk, sizeof(png_ihdr_chunk_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read IHDR chunk\n");
        fclose(file);
        return NULL;
    }

    ihdr_chunk.width = __builtin_bswap32(ihdr_chunk.width);
    ihdr_chunk.height = __builtin_bswap32(ihdr_chunk.height);

    // 跳过 CRC
    fseek(file, 4, SEEK_CUR);

    // 创建 image 结构体
    image_t* image = (image_t*)malloc(sizeof(image_t));
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image structure\n");
        fclose(file);
        return NULL;
    }

    image->name = strdup(filename);
    image->type = IMAGE_TYPE_PNG;
    image->raw = tensor_new(strdup(filename), TENSOR_TYPE_UINT8);

    // 确定颜色通道数
    int channels = 0;
    switch (ihdr_chunk.color_type) {
        case 0: channels = 1; break; // Grayscale
        case 2: channels = 3; break; // RGB
        case 3: channels = 1; break; // Indexed-color
        case 4: channels = 2; break; // Grayscale with alpha
        case 6: channels = 4; break; // RGBA
        default:
            fprintf(stderr, "Unsupported color type %d\n", ihdr_chunk.color_type);
            free(image->raw);
            free(image);
            fclose(file);
            return NULL;
    }

    // 简化处理：假设图像数据没有被压缩
    uint32_t img_size = ihdr_chunk.width * ihdr_chunk.height * channels;
    unsigned char* data = (unsigned char*)malloc(img_size);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for PNG data\n");
        free(image->raw);
        free(image);
        fclose(file);
        return NULL;
    }

    // 读取图像数据 (假设图像数据未压缩)
    while (fread(&chunk_header, sizeof(png_chunk_header_t), 1, file) == 1) {
        chunk_header.length = __builtin_bswap32(chunk_header.length);
        chunk_header.type = __builtin_bswap32(chunk_header.type);
        if (chunk_header.type == 0x49444154) { // IDAT
            if (fread(data, 1, chunk_header.length, file) != chunk_header.length) {
                fprintf(stderr, "Failed to read IDAT data\n");
                free(data);
                free(image->raw);
                free(image);
                fclose(file);
                return NULL;
            }
            fseek(file, 4, SEEK_CUR); // 跳过 CRC
            break;
        } else {
            fseek(file, chunk_header.length + 4, SEEK_CUR); // 跳过数据块和 CRC
        }
    }

    fclose(file);

    int dims[4] = {
        1,
        channels,
        ihdr_chunk.height,
        ihdr_chunk.width,
    };
    tensor_reshape(image->raw, 4, dims);
    tensor_apply(image->raw, data, img_size);

    return image;
}