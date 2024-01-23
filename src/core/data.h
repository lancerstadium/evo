

#ifndef CORE_DATA_TYPE_H
#define CORE_DATA_TYPE_H

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include "../util/util.h"


typedef enum {
    DATA_TYPE_FLAG_IS_SIGNED = 0b00000001,
    DATA_TYPE_FLAG_IS_STATIC = 0b00000010
} DataTypeFlag;

typedef enum {
    DATA_TYPE_U8,
    DATA_TYPE_U16,
    DATA_TYPE_U32,
    DATA_TYPE_U64,
    DATA_TYPE_I8,
    DATA_TYPE_I16,
    DATA_TYPE_I32,
    DATA_TYPE_I64,
    DATA_TYPE_F32,
    DATA_TYPE_F64,
    DATA_TYPE_CHAR,
    DATA_TYPE_BOOL,
    DATA_TYPE_SHORT,
    DATA_TYPE_INT,
    DATA_TYPE_LONG,
    DATA_TYPE_LONG_LONG,
    DATA_TYPE_UINT,
    DATA_TYPE_ULONG,
    DATA_TYPE_ULONG_LONG,
    DATA_TYPE_FLOAT,
    DATA_TYPE_DOUBLE,
    DATA_TYPE_STR,
    DATA_TYPE_USER_DEFINED,
    DATA_TYPE_ANY,
} DataTypeEnum;

typedef struct {
    DataTypeFlag flags;
    DataTypeEnum type;          // 标识类型
    const char* type_str;   // 该类型的等价字符串：不包括无符号或有符号关键字
} DataType;

#endif // CORE_DATA_TYPE_H