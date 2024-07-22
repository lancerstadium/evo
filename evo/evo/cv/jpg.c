#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../evo.h"

typedef struct jpeg_file_header jpeg_file_header_t;
typedef struct jpeg_frame_header jpeg_frame_header_t;


typedef struct {
    uint16_t marker;    // Marker (e.g., SOI marker is 0xFFD8)
    uint16_t length;    // Length of segment
    uint8_t data[1];    // Segment data (variable length)
} jpeg_segment_t;

EVO_PACKED(struct jpeg_file_header {
    uint16_t marker;     // Start of Image (SOI) marker (0xFFD8)
    uint16_t app0;       // Application marker (0xFFE0)
    uint16_t length;     // Length of APP0 field
    char identifier[5];  // "JFIF\0"
    uint8_t version[2];  // JFIF version
    uint8_t units;       // Density units
    uint16_t xdensity;   // X density
    uint16_t ydensity;   // Y density
    uint8_t xthumbnail;  // Thumbnail width
    uint8_t ythumbnail;  // Thumbnail height
});

EVO_PACKED(struct jpeg_frame_header {
    uint16_t marker;      // SOF0 marker (0xFFC0)
    uint16_t length;      // Length of the frame header
    uint8_t precision;    // Sample precision
    uint16_t height;      // Image height
    uint16_t width;       // Image width
    uint8_t ncomponents;  // Number of components (e.g., 3 for RGB, 4 for CMYK)
});

uint16_t read_uint16(FILE* file) {
    uint16_t value;
    if (fread(&value, sizeof(uint16_t), 1, file) != 1) {
        return 0;
    }
    return __builtin_bswap16(value);
}

image_t* image_load_jpg(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    jpeg_file_header_t file_header;
    if (fread(&file_header, sizeof(jpeg_file_header_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read JPEG file header\n");
        fclose(file);
        return NULL;
    }

    if (file_header.marker != 0xD8FF || file_header.app0 != 0xE0FF) {
        fprintf(stderr, "Not a valid JPEG file %s\n", filename);
        fclose(file);
        return NULL;
    }

    // Skip remaining bytes of APP0 segment
    fseek(file, file_header.length - 16, SEEK_CUR);

    jpeg_frame_header_t frame_header;
    while (fread(&frame_header.marker, sizeof(uint16_t), 1, file) == 1) {
        frame_header.marker = __builtin_bswap16(frame_header.marker);
        if (frame_header.marker == 0xFFC0) {
            // Read the frame header
            if (fread(&frame_header.length, sizeof(jpeg_frame_header_t) - sizeof(uint16_t), 1, file) != 1) {
                fprintf(stderr, "Failed to read JPEG frame header\n");
                fclose(file);
                return NULL;
            }
            frame_header.length = __builtin_bswap16(frame_header.length);
            frame_header.height = __builtin_bswap16(frame_header.height);
            frame_header.width = __builtin_bswap16(frame_header.width);
            break;
        } else {
            // Skip this segment
            uint16_t segment_length;
            if (fread(&segment_length, sizeof(uint16_t), 1, file) != 1) {
                fprintf(stderr, "Failed to read segment length\n");
                fclose(file);
                return NULL;
            }
            segment_length = __builtin_bswap16(segment_length);
            fseek(file, segment_length - 2, SEEK_CUR);
        }
    }

    if (frame_header.marker != 0xFFC0) {
        fprintf(stderr, "SOF0 marker not found in JPEG file %s\n", filename);
        fclose(file);
        return NULL;
    }

    // 创建 image 结构体
    image_t* image = (image_t*)malloc(sizeof(image_t));
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image structure\n");
        fclose(file);
        return NULL;
    }

    image->name = strdup(filename);
    image->type = IMAGE_TYPE_JPG;
    image->raw = tensor_new(strdup(filename), TENSOR_TYPE_UINT8);
    if (!image->raw) {
        fprintf(stderr, "Failed to allocate memory for tensor structure\n");
        free(image);
        fclose(file);
        return NULL;
    }

    uint32_t img_size = frame_header.width * frame_header.height * frame_header.ncomponents;

    unsigned char* data = (unsigned char*)malloc(img_size);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for JPEG data\n");
        image_free(image);
        fclose(file);
        return NULL;
    }

    if (fread(data, 1, img_size, file) != img_size) {
        fprintf(stderr, "Failed to read JPEG data\n");
        free(data);
        image_free(image);
        fclose(file);
        return NULL;
    }

    fclose(file);

    int dims[4] = {
        1,
        frame_header.height,
        frame_header.width,
        frame_header.ncomponents
    };
    
    tensor_reshape(image->raw, 4, dims);
    tensor_apply(image->raw, data, img_size);

    return image;
}