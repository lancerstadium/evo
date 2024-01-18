

#ifndef CORE_ETYPE_H
#define CORE_ETYPE_H

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include "../util/util.h"

typedef enum {
    ETYPE_U8,
    ETYPE_U16,
    ETYPE_U32,
    ETYPE_U64,
    ETYPE_I8,
    ETYPE_I16,
    ETYPE_I32,
    ETYPE_I64,
    ETYPE_F32,
    ETYPE_F64,
    ETYPE_BOOL,
    ETYPE_CHAR,
    ETYPE_STR,
    ETYPE_NONE
} EType;

typedef struct {
    EType type;
    union {
        u8 u8_v;
        u16 u16_v;
        u32 u32_v;
        u64 u64_v;
        i8 i8_v;
        i16 i16_v;
        i32 i32_v;
        i64 i64_v;
        f32 f32_v;
        f64 f64_v;
        bool bool_v;
        char char_v;
        Str *str_v;
        void *none_v;
    } value;
} EValue;

ALLOC_DEC_TYPE(EValue);

#endif // CORE_ETYPE_H