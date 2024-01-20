

// ==================================================================================== //
//                                    utils: str
// ==================================================================================== //

#ifndef UTIL_STR_H
#define UTIL_STR_H

#include "gtype.h"
#include "alloc.h"

typedef struct str {
    char* s;
    u32 len;
} Str;

ALLOC_DEC_TYPE(Str)


#define STR_EQ(str1, str2) \
    strcmp(str1, str2) == 0


// 判断c是否是delim
bool char_is_delim(char c, const char *delims);

char* char_display(char c);

// ==================================================================================== //
//                                    utils API: str
// ==================================================================================== //

Str* str_new(char* s);

Str* str_plus(Str* a, Str* b);

Str* str_plus_char(Str* a, char c);

void str_free(Str* p);

// 判断字符串是否以q开头
bool str_start_with(char* p, char* q);



#endif // UTIL_STR_H