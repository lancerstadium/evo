

#ifndef EVOC_HINT_H
#define EVOC_HINT_H

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "../utils/utils.h"

// ==================================================================================== //
//                                    evoc: error hint
// ==================================================================================== //

// ==================================================================================== //
//                                    Data: user input
// ==================================================================================== //

char *user_input;           // 用户输入字符串

// 编译器错误打印
void evoc_err(char *loc, char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);                          // 初始化ap
    int pos = loc - user_input;                 // 错误位置
    fprintf(stderr, "%s\n", user_input);        // 打印用户输入字符串
    fprintf(stderr, "%*s", pos, " ");           // 打印错误位置
    fprintf(stderr, "^ ");                      // 打印错误位置
    vfprintf(stderr, fmt, ap);                  // 打印错误信息
    va_end(ap);                                 // 释放ap
    fprintf(stderr,"\n");                       // 换行
}
// 判断字符串是否以q开头
bool str_start_with(char* p, char* q) {
    return strncmp(p, q, strlen(q)) == 0;
}


#endif // EVOC_HINT_H