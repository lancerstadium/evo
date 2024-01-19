

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
    ETYPE_INT,
    ETYPE_LONG,
    ETYPE_LONG_LONG,
    ETYPE_UINT,
    ETYPE_ULONG,
    ETYPE_ULONG_LONG,
    ETYPE_BOOL,
    ETYPE_CHAR,
    ETYPE_STR,
    ETYPE_ANY
} EType;

typedef struct {
    EType ty;       // evo 数值类型
    union {
        u8 u8val;
        u16 u16val;
        u32 u32val;
        u64 u64val;
        i8 i8val;
        i16 i16val;
        i32 i32val;
        i64 i64val;
        f32 f32val;
        f64 f64val;
        int ival;
        long lval;
        long long llval;
        unsigned int uival;
        unsigned long ulval;
        unsigned long long ullval;
        bool bval;
        char cval;
        const char *sval;
        void *anyval;
    } val;          // evo 数值
} EVal;

#endif // CORE_ETYPE_H