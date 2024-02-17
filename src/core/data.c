

#include "data.h"


/** 数据类型 */
const char* datatype_str[] = {
    [DATA_TYPE_U8]          = "u8", 
    [DATA_TYPE_U16]         = "u16", 
    [DATA_TYPE_U32]         = "u32", 
    [DATA_TYPE_U64]         = "u64", 
    [DATA_TYPE_I8]          = "i8",  
    [DATA_TYPE_I16]         = "i16", 
    [DATA_TYPE_I32]         = "i32", 
    [DATA_TYPE_I64]         = "i64",
    [DATA_TYPE_F32]         = "f32", 
    [DATA_TYPE_F64]         = "f64", 
    [DATA_TYPE_CHAR]        = "char", 
    [DATA_TYPE_BOOL]        = "bool",
    [DATA_TYPE_SHORT]       = "short",
    [DATA_TYPE_INT]         = "int",
    [DATA_TYPE_LONG]        = "lint",
    [DATA_TYPE_LONG_LONG]   = "llint",
    [DATA_TYPE_UINT]        = "uint",
    [DATA_TYPE_ULONG]       = "ulint",
    [DATA_TYPE_ULONG_LONG]  = "ullint",
    [DATA_TYPE_FLOAT]       = "float",
    [DATA_TYPE_DOUBLE]      = "double",
    [DATA_TYPE_STR]         = "str",
    [DATA_TYPE_STRUCT]      = "struct",
    [DATA_TYPE_UNION]       = "union",
    [DATA_TYPE_DEFINED]     = "defined",
    [DATA_TYPE_ANY]         = "any",
    [DATA_TYPE_NONE]        = "none"
};

const int datatype_str_num = GET_ARR_LEN(datatype_str);

int get_datatype_idx(const char* str) {
    int idx;
    for(idx = 0; idx < datatype_str_num; idx++) {
        if(STR_EQ(datatype_str[idx], str)) {
            return idx;
        }
    }
    return DATA_TYPE_UNKNOW;
}