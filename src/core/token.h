
#ifndef CORE_TOKEN_H
#define CORE_TOKEN_H


#include "eval.h"

typedef enum {
    TOKEN_TYPE_IDENTIFIER,      // 标识符
    TOKEN_TYPE_KEYWORD,         // 关键字
    TOKEN_TYPE_OPERATOR,        // 运算符
    TOKEN_TYPE_SYMBOL,          // 符号
    TOKEN_TYPE_NUMBER,          // 数字
    TOKEN_TYPE_STRING,          // 字符串
    TOKEN_TYPE_COMMENT,         // 注释
    TOKEN_TYPE_NEWLINE          // 换行
} token_type;

struct token {
    token_type type;
    int flags;
    
    union {
        char cval;
        const char* sval;
        unsigned int inum;
        unsigned long lnum;
        unsigned long long llnum;
        void* any;
    };

    bool whitespace;
    const char* between_brackets;   // 在括号中间的内容
    Pos pos;
};

struct token* token_create(struct token* _token);


#endif // CORE_TOKEN_H