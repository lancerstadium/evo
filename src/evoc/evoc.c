#include <ctype.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// ==================================================================================== //
//                                    evoc: lexer
// ==================================================================================== //

// ==================================================================================== //
//                                    Data: token
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
    Token head;                             // 令牌链表头
    head.next = NULL;                       // 初始化
    Token *cur = &head;                     // 当前令牌

    while(*p) {
        if(isspace(*p)) {                   // 如果是空格，跳过
            p++;
            continue;
        }
        if(strchr("+-*/()", *p)) {        // 如果是加号或减号
            cur = token_new(TK_RESERVED, cur, p++);
            continue;
        }
        if(isdigit(*p)) {                   // 如果是数字
            cur = token_new(TK_NUM, cur, p);
            cur->val = strtol(p, &p, 10);
            continue;
        }
        log_error("invalid token: %c", *p); break;
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
//                                     evoc: parser
// ==================================================================================== //

// ==================================================================================== //
//                                      Data: Node
// ==================================================================================== //

// parser语法分析：语法树节点类型
typedef enum {
    ND_ADD,                 // +
    ND_SUB,                 // -
    ND_MUL,                 // *
    ND_DIV,                 // /
    ND_NUM                  // 整数
} NodeType;

// parser语法分析：语法树节点
typedef struct Node Node;
struct Node {
    NodeType type;          // 节点类型
    Node *lhs, *rhs;        // 左右子节点
    int val;                // 节点值
};
// parser语法分析：创建新节点
Node* node_new(NodeType type) {
    Node *node = calloc(1, sizeof(Node));
    node->type = type;
    return node;
}
// parser语法分析：创建二元运算节点
Node* node_new_binary(NodeType type, Node *lhs, Node *rhs) {
    Node* node = node_new(type);
    node->lhs = lhs;
    node->rhs = rhs;
    return node;
}
// parser语法分析：创建数字节点
Node* node_new_num(int val) {
    Node* node = node_new(ND_NUM);
    node->val = val;
    return node;
}

// ==================================================================================== //
//                                    Parser: Expr
// ==================================================================================== //

// ==================================================================================== //
//                                     API: Expr
// ==================================================================================== //

Node* expr();
Node* mul();
Node* prim();
Node* unary();

// parser语法分析：表达式 `expr = mul ("+" mul | "-" mul)*`
Node* expr() {
    Node *node = mul();
    while(true) {
        if(consume('+')) {
            node = node_new_binary(ND_ADD, node, mul());
        }else if(consume('-')) {
            node = node_new_binary(ND_SUB, node, mul());
        }else {
            return node;
        }
    }
}
// parser语法分析：表达式 `mul = unary ("*" unary | "/" unary)*`
Node* mul() {
    Node *node = unary();
    while(true) {
        if(consume('*')) {
            node = node_new_binary(ND_MUL, node, unary());
        }else if(consume('/')) {
            node = node_new_binary(ND_DIV, node, unary());
        }else {
            return node;
        }
    }
}
// parser语法分析：表达式 `prim = num | "(" expr ")"`
Node* prim() {
    if(consume('(')) {
        Node *node = expr();
        expect(')');
        return node;
    }
    return node_new_num(expect_number());
}
// parser语法分析：表达式 `unary = ("+" | "-")? prim`
Node* unary() {
    if(consume('+')) {
        return prim();
    }else if(consume('-')) {
        return node_new_binary(ND_SUB, node_new_num(0), prim());
    }
    return prim();
}

// parser语法分析：生成汇编代码
void gen(Node *node) {
    if(node->type == ND_NUM) {                      // 数字节点：叶子节点
        printf("  push %d\n", node->val);           // 入栈：数字
        return;
    }
    gen(node->lhs);                                 // 左节点
    gen(node->rhs);                                 // 右节点
    printf("  pop rdi\n");                          // 出栈：右操作数
    printf("  pop rax\n");                          // 出栈：左操作数
    
    switch(node->type) {
        case ND_ADD:                                // 如果为`+`
            printf("  add rax, rdi\n"); break;      // rax += rdi
        case ND_SUB:                                // 如果为`-`
            printf("  sub rax, rdi\n"); break;      // rax -= rdi
        case ND_MUL:                                // 如果为`*`
            printf("  imul rax, rdi\n"); break;     // rax *= rdi
        case ND_DIV:                                // 如果为`/`
            printf("  cqo\n");                      // cqo ：rax = rdx:rax
            printf("  idiv rdi\n");                 // rax /= rdi ... rdx
            break;
    }

    printf("  push rax\n");                         // 入栈：结果
}


// ==================================================================================== //
//                                  Proc Entry: evoc
// ==================================================================================== //


int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "usage: %s <expr>\n", argv[0]);
        return 1;
    }

    // 解析用户输入
    user_input = argv[1];                                       // 保存用户输入
    token = token_identify(user_input);                         // 识别表达式
    Node *node = expr();                                        // 解析表达式

    // 生成汇编代码前面部分
    printf(".intel_syntax noprefix\n");                         // 设置汇编语法格式：INTEL，无前缀
    printf(".globl main\n");                                    // 定义全局变量：main
    printf("main:\n");                                          // 定义main函数

    gen(node);                                                  // 生成汇编代码
    printf("  pop rax\n");                                      // 出栈：计算结果
    printf("  ret\n");                                          // 返回
    return 0;
}