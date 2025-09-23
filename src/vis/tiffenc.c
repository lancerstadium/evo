#include "tiffenc.h"

#include <stdio.h>

#define TIFF_IFD_SIZE 12

void tiff_ifh(FILE* file) {
    IFH ifh;
    ifh.endian = 0x4949; // 小端
    ifh.magic = 42; // TIFF魔数
    ifh.ifdOffset = sizeof(IFH); // IFD偏移量
    fwrite(&ifh, sizeof(IFH), 1, file);
}

// 写入IFD信息
void tiff_ifd(FILE *file, imgInfo* info) {
    // 这里需要根据实际的IFD信息来写入，以下为示例
    uint16_t entries = 10; // 假设有7个目录项
    fwrite(&entries, sizeof(uint16_t), 1, file);
    // 写入每个目录项，这里省略具体实现
    uint16_t tag,type,nbit[]={info->bits,info->bits,info->bits,info->bits};
    uint32_t count,value,nextIFD=0;
    //0100,图像宽度
    tag=256;    type=4;     count=1;   value=info->width;   
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0101,图像高度
    tag=257;    value=info->height;                         
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0102,像素颜色通道的bit数组位置(距文件开头的偏移量)
    tag=258;    type=3;     count=4;    value=8+2+TIFF_IFD_SIZE*entries+sizeof(nextIFD); 
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0103,图像压缩方案,=1表示无压缩
    tag=259;    count=1;    value=1;                       
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0106,颜色空间,=5表示CMYK
    tag=262;    value=5;                                   
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0111,每个Strip的位置(距文件开头的偏移量)
    tag=273;    type=4;    value=8+2+TIFF_IFD_SIZE*entries+sizeof(nextIFD)+sizeof(nbit);       
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0115,每个像素的通道数,CMYK为4
    tag=277;    type=3;     value=4;                       
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0116,每个Strip的行数
    tag=278;    type=4;     value=info->height;             
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //0117,每个Strip的长度
    tag=279;    value=info->height*info->width*4;            
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    //011C,像素通道存储方式,=1表示CMYKCMYK...,=2表示CC...MM...YY...KK...
    tag=284;    type=3;     value=1;                       
    fwrite(&tag, sizeof(tag), 1, file);
    fwrite(&type, sizeof(type), 1, file);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&value, sizeof(value), 1, file);
    fwrite(&nextIFD, sizeof(nextIFD), 1, file);
    fwrite(&nbit, sizeof(nbit), 1, file);
}
// 将数组保存为TIFF文件
void tiff_encode(uint8_t* array, const char* filename, uint32_t width, uint32_t height, uint16_t channel, uint16_t bitsPerSample) {
    FILE* file = fopen(filename,"wb");
    if (!file) 
        perror("无法打开文件");
    imgInfo info = {width, height, channel, bitsPerSample};
    // 写入文件头
    tiff_ifh(file);
    // 写入IFD信息
    tiff_ifd(file, &info);
    // 写入图像数据
    fwrite(array, width*height*channel*sizeof(uint8_t),1,file);
}