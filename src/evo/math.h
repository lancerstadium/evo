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
//                                       evo/math.h
// ==================================================================================== //

#ifndef __EVO_MATH_H__
#define __EVO_MATH_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ==================================================================================== //
//                                       macros
// ==================================================================================== //

#if defined(__GNUC__) || defined(__clang__)
#define UNUSED      __attribute__((unused))
#define EXPORT      __attribute__((visibility("default")))
#define PACKED(D)   D __attribute__((packed))
#define NORETURN    __attribute__((noreturn))
#elif defined(MSC_VER)
#define UNUSED      __pragma(warning(suppress:4100))
#define EXPORT      __pragma(warning(suppress:4091))
#define PACKED(D)   __pragma(pack(push, 1)) D __pragma(pack(pop))
#define NORETURN
#else   // __GNUC__ || __clang__
#define UNUSED
#define EXPORT
#define PACKED(D)   D
#define NORETURN
#endif  // __GNUC__ || __clang__


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

#else   // __ARM_ARCH || __riscv

typedef struct fp16_pack __fp16;

PACKED(struct fp16_pack {
    unsigned short frac : 10;
    unsigned char exp : 5;
    unsigned char sign : 1;
});


PACKED(struct fp32_pack {
    unsigned int frac : 23;
    unsigned char exp : 8;
    unsigned char sign : 1;
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



#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __EVO_MATH_H__