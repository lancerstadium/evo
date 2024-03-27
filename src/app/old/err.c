
// ==================================================================================== //
//                                    evoc: error hint
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                     Pub API: error
// ==================================================================================== //

// 编译器错误打印：打印错误信息
void evoc_err(char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);                          // 初始化ap
    fprintf(stderr, "%s\n", current_input);     // 打印用户输入字符串
    vfprintf(stderr, fmt, ap);                  // 打印错误信息
    va_end(ap);                                 // 释放ap
    fprintf(stderr,"\n");                       // 换行
    exit(1);                                    // 退出
}
// 编译器错误打印：打印字符位置
void evoc_verr_at(char *loc, char *fmt, va_list ap) {
    int position = loc - current_input;         // 错误位置
    fprintf(stderr, "%s\n", current_input);     // 打印用户输入字符串
    fprintf(stderr, "%*s", position, "");       // 打印错误位置
    fprintf(stderr, "^ ");                      // 打印错误位置
    vfprintf(stderr, fmt, ap);                  // 打印错误信息
    fprintf(stderr, "\n");                      // 换行
    exit(1);                                    // 退出
}
// 编译器错误打印：打印字符位置
void evoc_err_at(char *loc, char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);                          // 初始化ap
    evoc_verr_at(loc, fmt, ap);                 // 打印错误信息
    va_end(ap);                                 // 释放ap
}
// 编译器错误打印：打印Token
void evoc_err_tok(Token *tok, char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);                          // 初始化ap
    evoc_verr_at(tok->loc, fmt, ap);            // 打印错误信息
    va_end(ap);                                 // 释放ap
}