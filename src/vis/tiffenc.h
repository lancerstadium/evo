#ifndef TIFFENC_H
#define TIFFENC_H

#include <stdint.h>

typedef struct {
    uint16_t endian;
    uint16_t magic;
    uint32_t ifdOffset;
} IFH;

typedef struct {
    uint32_t width; // 图像宽度
    uint32_t height; // 图像高度
    uint16_t channel; // 图像通道数
    uint16_t bits; // 每个样本的位数
} imgInfo;


void tiff_encode(uint8_t* array, const char* filename, uint32_t width, uint32_t height, uint16_t channel, uint16_t bitsPerSample);


#endif