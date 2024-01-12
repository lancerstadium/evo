#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../utils/utils.h"


// ==================================================================================== //
//                                    evoc: lexer
// ==================================================================================== //

// lexer词法分析：识别令牌
typedef enum {
    TK_NUM,                 // 数字令牌
    TK_PLUS,                // 加号令牌
    TK_MINUS,               // 减号令牌
    TK_LPAREN,              // 左括号令牌
    TK_RPAREN,              // 右括号令牌
    TK_RESERVED,            // 保留字令牌
    TK_END                  // 结束令牌
} TokenType;

// ==================================================================================== //
//                                    Data: token
// ==================================================================================== //

// lexer词法分析：令牌结构体
typedef struct Token Token;
struct Token {
    TokenType type;         // 令牌类型
    Token *next;            // 下一个令牌
    int val;                // 令牌值
    char *str;              // 令牌字符串
};

Token *token;               // 现在的令牌

// lexer词法分析：创建新令牌
Token* token_new(TokenType type, Token *cur, char *str) {
    Token *tok = calloc(1, sizeof(Token));
    tok->type = type;
    tok->str = str;
    cur->next = tok;
    return tok;
}

// lexer词法分析：识别令牌
Token* token_identify(char *p) {
    Token head;             // 令牌链表头
    head.next = NULL;       // 初始化
    Token *cur = &head;     // 当前令牌

    while(*p) {
        if(isspace(*p)) {   // 如果是空格，跳过
            p++;
            continue;
        }
        if(*p == '+' || *p == '-') { // 如果是加号或减号
            cur = token_new(TK_RESERVED, cur, p++);
            continue;
        }
        if(isdigit(*p)) {   // 如果是数字
            cur = token_new(TK_NUM, cur, p);
            cur->val = strtol(p, &p, 10);
            continue;
        }
        if(*p == '(') {     // 如果是左括号
            cur = token_new(TK_LPAREN, cur, p++);
            continue;
        }
        if(*p == ')') {     // 如果是右括号
            cur = token_new(TK_RPAREN, cur, p++);
            continue;
        }
        log_error("can't identify: %c", *p); break;
    }
    cur = token_new(TK_END, cur, p);
    return head.next;
}

// ==================================================================================== //
//                                   Token Operations
// ==================================================================================== //


// lexer词法分析：尝试消费下一个令牌
bool consume(char op) {
    if(token->type == TK_END || token->str[0] != op) {
        // 如果是结束令牌，或者不是期望的令牌，返回false
        // log_error("expect `%c`, but got `%c`", op, token->str[0]);
        return false;
    }
    // 否则，token下移，返回true
    token = token->next;
    return true;
}
// lexer词法分析：期望下一个令牌
void expect(char op) {
    if(token->type != TK_RESERVED || token->str[0] != op) {
        log_error("expect `%c`, but got `%c`", op, token->str[0]);
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




// ==================================================================================== //
//                                  Proc Entry: evoc
// ==================================================================================== //


int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "usage: %s <expr>\n", argv[0]);
        return 1;
    }

    // 识别argv[1]中的表达式
    token = token_identify(argv[1]);

    printf(".intel_syntax noprefix\n");                         // 设置汇编语法格式：INTEL，无前缀
    printf(".globl main\n");                                    // 定义全局变量：main
    printf("main:\n");                                          // 定义main函数
    printf("  mov rax, %d\n", expect_number());                 // mov rax, argv[1]

    while(!at_end()) {                                          // 循环直到到达词法末尾
        if(consume('+')) {                                      // 加号：+
            printf("  add rax, %d\n", expect_number());         // add rax, argv[3]
            continue;
        }
        expect('-');                                            // 减号：-
        printf("  sub rax, %d\n", expect_number());             // sub rax, argv[5]
    }
    printf("  ret\n");                                          // 返回
    return 0;
}