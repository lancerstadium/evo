


#ifndef EVOC_LEXER_H
#define EVOC_LEXER_H

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include <ctype.h>
#include <stdlib.h>
#include <stdbool.h>
#include "hint.h"

// ==================================================================================== //
//                                    evoc: lexer
// ==================================================================================== //

// ==================================================================================== //
//                                    Data: Token
// ==================================================================================== //

// lexer词法分析：识别令牌
typedef enum {
    TK_NUM,                 // 数字令牌
    TK_RESERVED,            // 保留字令牌
    TK_END                  // 结束令牌
} TokenType;
// lexer词法分析：令牌结构体
typedef struct Token Token;
struct Token {
    TokenType type;                         // 令牌类型
    Token *next;                            // 下一个令牌
    int val;                                // 令牌值
    char *str;                              // 令牌字符串
    int len;                                // 令牌字符串长度
};
Token *token;                               // 现在的令牌

// lexer词法分析：创建新令牌
Token* token_new(TokenType type, Token *cur, char *str, int len) {
    Token *tok = (Token*)calloc(1, sizeof(Token));
    tok->type = type;
    tok->str = str;
    tok->len = len;
    cur->next = tok;
    return tok;
}

// lexer词法分析：识别令牌
Token* token_identify(char *p) {
    Token head;                             // 令牌链表头
    head.next = NULL;                       // 初始化
    Token *cur = &head;                     // 当前令牌

    while(*p) {
        // 如果是空格，跳过
        if(isspace(*p)) {                  
            p++;
            continue;
        }
        // 如果是多长度运算符
        if(str_start_with(p, "==") || 
           str_start_with(p, "!=") ||
           str_start_with(p, "<=") || 
           str_start_with(p, ">=")) {          
            cur = token_new(TK_RESERVED, cur, p, 2);
            p += 2;
            continue;
        }
        // 如果是单长度运算符
        if(strchr("+-*/()<>", *p)) {
            cur = token_new(TK_RESERVED, cur, p++, 1);
            continue;
        }
        // 如果是数字
        if(isdigit(*p)) {                   
            cur = token_new(TK_NUM, cur, p, 0);
            char *q = p;
            cur->val = strtol(p, &p, 10);
            cur->len = p - q;
            continue;
        }
        // log_error("invalid token: %c", *p); break;
        evoc_err(p, "invalid token: %c", *p); break;
    }
    cur = token_new(TK_END, cur, p, 0);
    return head.next;
}

// ==================================================================================== //
//                                   Token Operations
// ==================================================================================== //


// lexer词法分析：尝试消费下一个令牌
bool consume(char* op) {
    if(token->type == TK_END || strlen(op) != token->len || memcmp(token->str, op, token->len)) {
        // 如果是结束令牌，或者不是期望的令牌，返回false
        // log_error("expect `%c`, but got `%c`", op, token->str[0]);
        return false;
    }
    // 否则，token下移，返回true
    token = token->next;
    return true;
}
// lexer词法分析：期望下一个令牌
void expect(char* op) {
    if(token->type != TK_RESERVED || strlen(op) != token->len || memcmp(token->str, op, token->len)) {
        // log_error("expect `%c`, but got `%c`", op, token->str[0]);
        evoc_err(token->str, "expect `%c`", op);
    }
    token = token->next;
}

// lexer词法分析：获取当前令牌的数字值
int expect_number() {
    if(token->type != TK_NUM) {
        log_error("expect number");
    }
    int val = token->val;
    token = token->next;
    return val;
}

// lexer词法分析：到达令牌末尾
bool at_end() {
    return token->type == TK_END;
}

#endif // EVOC_LEXER_H