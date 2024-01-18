

// ==================================================================================== //
//                                     util: utf-8
// ==================================================================================== //

#include "utf8.h"


// ==================================================================================== //
//                                    Pub API: utf-8
// ==================================================================================== //

/**
 * @brief 获取UTF-8编码长度
 * @param value 字符值
 * @return u32 编码长度
 */
u32 utf8_get_encode_len(u32 value) {
    if(value <= 0x7f) {
        return 1;
    } else if(value <= 0x7ff) {
        return 2;
    } else if(value <= 0xffff) {
        return 3;
    } else if(value <= 0x1fffff) {
        return 4;
    } else {
        return 0;
    }
}

/**
 * @brief 获取UTF-8解码长度
 * @param byte UTF-8的最高1字节
 * @return u32 解码长度
 */
u32 utf8_get_decode_len(u8 byte) {
    if((byte & 0x80) == 0x00) {
        return 1;
    } else if((byte & 0xe0) == 0xc0) {
        return 2;
    } else if((byte & 0xf0) == 0xe0) {
        return 3;
    } else if((byte & 0xf8) == 0xf0) {
        return 4;
    } else {
        return 0;
    }
}

/**
 * @brief UTF-8编码
 * @param buf 字符缓冲
 * @param value 字符值
 * @return u32编码长度
 */
u32 utf8_encode(u8* buf, u32 value) {
    if(value <= 0x7f) {
        buf[0] = value;
        return 1;
    } else if(value <= 0x7ff) {
        buf[0] = 0xc0 | (value >> 6);
        buf[1] = 0x80 | (value & 0x3f);
        return 2;
    } else if(value <= 0xffff) {
        buf[0] = 0xe0 | (value >> 12);
        buf[1] = 0x80 | ((value >> 6) & 0x3f);
        buf[2] = 0x80 | (value & 0x3f);
        return 3;
    } else if(value <= 0x1fffff) {
        buf[0] = 0xf0 | (value >> 18);
        buf[1] = 0x80 | ((value >> 12) & 0x3f);
        buf[2] = 0x80 | ((value >> 6) & 0x3f);
        buf[3] = 0x80 | (value & 0x3f);
        return 4;
    }
    return 0;
}

/**
 * @brief UTF-8解码
 * @param buf 字符缓冲
 * @param len 编码长度
 * @return u32 字符值
 */
u32 utf8_decode(u8* buf, u32 len) {
    if(len == 1) {
        return buf[0];
    } else if(len == 2) {
        return ((buf[0] & 0x1f) << 6) | (buf[1] & 0x3f);
    } else if(len == 3) {
        return ((buf[0] & 0x0f) << 12) | ((buf[1] & 0x3f) << 6) | (buf[2] & 0x3f);
    } else if(len == 4) {
        return ((buf[0] & 0x07) << 18) | ((buf[1] & 0x3f) << 12) | ((buf[2] & 0x3f) << 6) | (buf[3] & 0x3f);
    }
    return 0;
}