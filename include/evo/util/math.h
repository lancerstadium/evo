/**
 * =================================================================================== //
 * @file math.h
 * @author lancerstadium (lancerstadium@163.com)
 * @brief math header file
 * @version 0.1
 * @date 2024-07-03
 * @copyright Copyright (c) 2024
 * =================================================================================== //
 */

// ==================================================================================== //
//                                    evo/util/math.h
// ==================================================================================== //

#ifndef __EVO_UTIL_MATH_H__
#define __EVO_UTIL_MATH_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       macros
// ==================================================================================== //

#if defined(__GNUC__) || defined(__clang__)
#define UNUSED __attribute__((unused))
#define EXPORT __attribute__((visibility("default")))
#define PACKED(D) D __attribute__((packed))
#define NORETURN __attribute__((noreturn))
#elif defined(MSC_VER)
#define UNUSED __pragma(warning(suppress : 4100))
#define EXPORT __pragma(warning(suppress : 4091))
#define PACKED(D) __pragma(pack(push, 1)) D __pragma(pack(pop))
#define NORETURN
#else  // __GNUC__ || __clang__
#define UNUSED
#define EXPORT
#define PACKED(D) D
#define NORETURN
#endif  // __GNUC__ || __clang__

// ==================================================================================== //
//                                    calcu macros
// ==================================================================================== //

#define MIN(a, b) ({typeof(a) _amin = (a); typeof(b) _bmin = (b); (void)((void*)&_amin == (void*)&_bmin); _amin < _bmin ? _amin : _bmin; })
#define MAX(a, b) ({typeof(a) _amax = (a); typeof(b) _bmax = (b); (void)((void*)&_amax == (void*)&_bmax); _amax > _bmax ? _amax : _bmax; })
#define CLAMP(v, a, b) MIN(MAX(a, v), b)

// ==================================================================================== //
//                                      swab
// ==================================================================================== //

static inline uint16_t __swab16(uint16_t x) {
    return ((x << 8) | (x >> 8));
}

static inline uint32_t __swab32(uint32_t x) {
    return ((x << 24) | (x >> 24) |
            ((x & (uint32_t)0x0000ff00UL) << 8) |
            ((x & (uint32_t)0x00ff0000UL) >> 8));
}

static inline uint64_t __swab64(uint64_t x) {
    return ((x << 56) | (x >> 56) |
            ((x & (uint64_t)0x000000000000ff00ULL) << 40) |
            ((x & (uint64_t)0x0000000000ff0000ULL) << 24) |
            ((x & (uint64_t)0x00000000ff000000ULL) << 8) |
            ((x & (uint64_t)0x000000ff00000000ULL) >> 8) |
            ((x & (uint64_t)0x0000ff0000000000ULL) >> 24) |
            ((x & (uint64_t)0x00ff000000000000ULL) >> 40));
}

static inline uint32_t __swahw32(uint32_t x) {
    return (((x & (uint32_t)0x0000ffffUL) << 16) | ((x & (uint32_t)0xffff0000UL) >> 16));
}

static inline uint32_t __swahb32(uint32_t x) {
    return (((x & (uint32_t)0x00ff00ffUL) << 8) | ((x & (uint32_t)0xff00ff00UL) >> 8));
}

// ==================================================================================== //
//                                      cpu
// ==================================================================================== //

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define cpu_to_le64(x) (__swab64((uint64_t)(x)))
#define le64_to_cpu(x) (__swab64((uint64_t)(x)))
#define cpu_to_le32(x) (__swab32((uint32_t)(x)))
#define le32_to_cpu(x) (__swab32((uint32_t)(x)))
#define cpu_to_le16(x) (__swab16((uint16_t)(x)))
#define le16_to_cpu(x) (__swab16((uint16_t)(x)))
#define cpu_to_be64(x) ((uint64_t)(x))
#define be64_to_cpu(x) ((uint64_t)(x))
#define cpu_to_be32(x) ((uint32_t)(x))
#define be32_to_cpu(x) ((uint32_t)(x))
#define cpu_to_be16(x) ((uint16_t)(x))
#define be16_to_cpu(x) ((uint16_t)(x))
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define cpu_to_le64(x) ((uint64_t)(x))
#define le64_to_cpu(x) ((uint64_t)(x))
#define cpu_to_le32(x) ((uint32_t)(x))
#define le32_to_cpu(x) ((uint32_t)(x))
#define cpu_to_le16(x) ((uint16_t)(x))
#define le16_to_cpu(x) ((uint16_t)(x))
#define cpu_to_be64(x) (__swab64((uint64_t)(x)))
#define be64_to_cpu(x) (__swab64((uint64_t)(x)))
#define cpu_to_be32(x) (__swab32((uint32_t)(x)))
#define be32_to_cpu(x) (__swab32((uint32_t)(x)))
#define cpu_to_be16(x) (__swab16((uint16_t)(x)))
#define be16_to_cpu(x) (__swab16((uint16_t)(x)))
#else  // Unknown
#define cpu_to_le64(x) ((uint64_t)(x))
#define le64_to_cpu(x) ((uint64_t)(x))
#define cpu_to_le32(x) ((uint32_t)(x))
#define le32_to_cpu(x) ((uint32_t)(x))
#define cpu_to_le16(x) ((uint16_t)(x))
#define le16_to_cpu(x) ((uint16_t)(x))
#define cpu_to_be64(x) (__swab64((uint64_t)(x)))
#define be64_to_cpu(x) (__swab64((uint64_t)(x)))
#define cpu_to_be32(x) (__swab32((uint32_t)(x)))
#define be32_to_cpu(x) (__swab32((uint32_t)(x)))
#define cpu_to_be16(x) (__swab16((uint16_t)(x)))
#define be16_to_cpu(x) (__swab16((uint16_t)(x)))
#endif


// ==================================================================================== //
//                                    string hash
// ==================================================================================== //

static inline unsigned int shash(const char* s) {
    unsigned int v = 5381;
    if (s) {
        while (*s)
            v = (v << 5) + v + (*s++);
    }
    return v;
}

// ==================================================================================== //
//                                     interger
// ==================================================================================== //

int imin(int a, int b);
int imax(int a, int b);
int min_abs(int a, int b);
int max_abs(int a, int b);
int gcd(int a, int b);
int lcm(int a, int b);
int align(int value, int step);
void* align_address(void* address, int step);

// ==================================================================================== //
//                                    float 16 & 32
// ==================================================================================== //

#if defined(__ARM_ARCH) || defined(__riscv)

#define fp16_to_fp32(data) \
    ({                     \
        float f = data;    \
        f;                 \
    })

#define fp32_to_fp16(data) \
    ({                     \
        __fp16 f = data;   \
        f;                 \
    })

#else  // __ARM_ARCH || __riscv

typedef struct fp16_pack __fp16;

PACKED(struct fp16_pack {
    unsigned int frac : 10;
    unsigned int exp : 5;
    unsigned int sign : 1;
});

PACKED(struct fp32_pack {
    unsigned int frac : 23;
    unsigned int exp : 8;
    unsigned int sign : 1;
});

static inline float fp16_to_fp32(__fp16 data) {
    float f;
    struct fp32_pack* fp32 = (struct fp32_pack*)&f;
    struct fp16_pack* fp16 = &data;
    int exp = fp16->exp;
    if (exp == 31 && fp16->frac != 0) {
        fp32->sign = fp16->sign;
        fp32->exp = 255;
        fp32->frac = 1;
        return f;
    }
    if (exp == 31)
        exp = 255;
    if (exp == 0)
        exp = 0;
    else
        exp = (exp - 15) + 127;
    fp32->exp = exp;
    fp32->sign = fp16->sign;
    fp32->frac = ((int)fp16->frac) << 13;

    return f;
}

static inline __fp16 fp32_to_fp16(float data) {
    struct fp32_pack* fp32 = (struct fp32_pack*)&data;
    struct fp16_pack fp16;
    int exp = fp32->exp;
    if (fp32->exp == 255 && fp32->frac != 0) {  // NaN
        fp16.exp = 31;
        fp16.frac = 1;
        fp16.sign = fp32->sign;
        return fp16;
    }
    if ((exp - 127) < -14)
        exp = 0;
    else if ((exp - 127) > 15)
        exp = 31;
    else
        exp = exp - 127 + 15;
    fp16.exp = exp;
    fp16.frac = fp32->frac >> 13;
    fp16.sign = fp32->sign;
    return fp16;
}

#endif  // __ARM_ARCH || __riscv

/*
 * float16, bfloat16 and float32 conversion
 */
static inline uint16_t float32_to_float16(float v) {
    union {
        uint32_t u;
        float f;
    } t;
    uint16_t y;

    t.f = v;
    y = ((t.u & 0x7fffffff) >> 13) - (0x38000000 >> 13);
    y |= ((t.u & 0x80000000) >> 16);
    return y;
}

static inline float float16_to_float32(uint16_t v) {
    union {
        uint32_t u;
        float f;
    } t;

    t.u = v;
    t.u = ((t.u & 0x7fff) << 13) + 0x38000000;
    t.u |= ((v & 0x8000) << 16);
    return t.f;
}

static inline uint16_t float32_to_bfloat16(float v) {
    union {
        uint32_t u;
        float f;
    } t;

    t.f = v;
    return t.u >> 16;
}

static inline float bfloat16_to_float32(uint16_t v) {
    union {
        uint32_t u;
        float f;
    } t;

    t.u = v << 16;
    return t.f;
}

// ==================================================================================== //
//                                    quantize
// ==================================================================================== //

void symmetric_quantize_float32_to_int16(float *input, int16_t *output, int size, float scale);
void symmetric_dequantize_int16_to_float32(int16_t *input, float *output, int size, float scale);
void symmetric_quantize_float32_to_int8(float *input, int8_t *output, int size, float scale);
void symmetric_dequantize_int8_to_float32(int8_t *input, float *output, int size, float scale);
void asymmetric_quantize_float32_to_int16(float *input, int16_t *output, int size, float scale, int16_t zero_point);
void asymmetric_dequantize_int16_to_float32(int16_t *input, float *output, int size, float scale, int16_t zero_point);
void asymmetric_quantize_float32_to_int8(float *input, int8_t *output, int size, float scale, int8_t zero_point);
void asymmetric_dequantize_int8_to_float32(int8_t *input, float *output, int size, float scale, int8_t zero_point);
void dynamic_range_quantize_float32_to_int16(float *input, int16_t *output, int size);
void dynamic_range_dequantize_int16_to_float32(int16_t *input, float *output, int size, float min, float max);
void dynamic_range_quantize_float32_to_int8(float *input, int8_t *output, int size);
void dynamic_range_dequantize_int8_to_float32(int8_t *input, float *output, int size, float min, float max);
void log_quantize_float32_to_int16(float *input, int16_t *output, int size);
void log_dequantize_int16_to_float32(int16_t *input, float *output, int size);
void log_quantize_float32_to_int8(float *input, int8_t *output, int size);
void log_dequantize_int8_to_float32(int8_t *input, float *output, int size);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_UTIL_MATH_H__