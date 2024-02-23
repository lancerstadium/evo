

#ifndef CORE_DATA_TYPE_H
#define CORE_DATA_TYPE_H

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include "../util/util.h"


typedef enum {
    DATA_TYPE_FLAG_IS_SIGNED = 0b00000001,
    DATA_TYPE_FLAG_IS_STATIC = 0b00000010,
    DATA_TYPE_FLAG_IS_CONST  = 0b00000100,
    DATA_TYPE_FLAG_IS_PTR    = 0b00001000
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
    DATA_TYPE_BYTE,     // u8
    DATA_TYPE_CHAR,
    DATA_TYPE_BOOL,
    DATA_TYPE_RUNE,     // i32
    DATA_TYPE_STR,
    DATA_TYPE_ANY,
    DATA_TYPE_VOID,
    DATA_TYPE_DEFINED,
    DATA_TYPE_UNKNOWN = -1
} DataTypeEnum;

typedef struct {
    int flags;
    DataTypeEnum type;          // 标识类型
    const char* type_str;       // 该类型的等价字符串：不包括无符号或有符号关键字
    size_t size;                // 数据大小
    int ptr_depth;              // 指针深度
} DataType;

extern const char* datatype_str[];
extern const int datatype_str_num;

int get_datatype_idx(const char* str);

#endif // CORE_DATA_TYPE_H