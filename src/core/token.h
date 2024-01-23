
#ifndef CORE_TOKEN_H
#define CORE_TOKEN_H


#include "data.h"

typedef enum {
    TOKEN_TYPE_EOF,             // 结束
    TOKEN_TYPE_IDENTIFIER,      // 标识符
    TOKEN_TYPE_KEYWORD,         // 关键字
    TOKEN_TYPE_OPERATOR,        // 运算符
    TOKEN_TYPE_SYMBOL,          // 符号
    TOKEN_TYPE_NUMBER,          // 数字
    TOKEN_TYPE_STRING,          // 字符串
    TOKEN_TYPE_COMMENT,         // 注释
    TOKEN_TYPE_NEWLINE          // 换行
} TokenType;

typedef struct token {
    TokenType type;
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
} Token;

extern Token tmp_token;


char* token_get_type_str(Token* tok);
void token_read(Token *tok);
Token* token_create(Token* _token);


#endif // CORE_TOKEN_H