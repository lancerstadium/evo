
#ifndef EVOC_EVOC_H
#define EVOC_EVOC_H

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#define _POSIX_C_SOURCE 200809L
#define EVOC_PARSER_MAX_NODE 1024

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include "../utils/utils.h"

// ==================================================================================== //
//                                       Define
// ==================================================================================== //




// ==================================================================================== //
//                                 evoc Pub Type: Token
// ==================================================================================== //

// lexer词法分析：识别令牌
typedef enum {
    TK_NUM,                                 // 数字令牌
    TK_IDENT,                               // 标识符令牌
    TK_PUNCT,                               // 运算符令牌
    TK_KEYWORD,                             // 关键字令牌
    TK_RESERVED,                            // 保留字令牌
    TK_EOF                                  // 结束令牌
} TokenType;
// lexer词法分析：令牌结构体
typedef struct Token Token;
struct Token {
    TokenType type;                         // 令牌类型
    Token *next;                            // 下一个令牌
    int val;                                // 令牌值
    char *loc;                              // 令牌字符串位置
    int len;                                // 令牌字符串长度
};

// ==================================================================================== //
//                                  evoc Pub Type: Var
// ==================================================================================== //

// var变量：变量结构体
typedef struct Var Var;
struct Var {
    char *name;                             // 变量名
    int offset;                             // 变量偏移
    Var *next;                              // 下一个变量
};

// ==================================================================================== //
//                                  evoc Pub Type: Node
// ==================================================================================== //

// parser语法分析：语法树节点类型
typedef enum {
    ND_ADD,                                 // +
    ND_SUB,                                 // -
    ND_MUL,                                 // *
    ND_DIV,                                 // /
    ND_NEG,                                 // unary -
    ND_EQU,                                 // ==
    ND_NEQ,                                 // !=
    ND_LSS,                                 // <
    ND_GTR,                                 // >
    ND_LEQ,                                 // <=
    ND_GEQ,                                 // >=
    ND_ASSIGN,                              // =
    ND_IF,                                  // `if`
    ND_LOOP,                                // `for` or `while`
    ND_RETURN,                              // `return`
    ND_BLOCK,                               // { ... }
    ND_VAR,                                 // variable
    ND_EXPR_STMT,                           // expression statement
    ND_NUM                                  // integer
} NodeType;

// parser语法分析：语法树节点
typedef struct Node Node;
struct Node {
    NodeType type;                          // 节点类型
    Node *lhs, *rhs;                        // 左右子节点
    Node* next;                             // 下一个节点                    
    // 节点值：ND_NUM
    int val;
    // 变量：ND_VAR                                
    Var *var;   
    // 代码块主体：ND_BLOCK
    Node* body;  
    // `if`或`for`语句：ND_IF、ND_FOR
    Node* cond, *then, *els, *init, *inc;   

};

// ==================================================================================== //
//                                  evoc Pub Type: Func
// ==================================================================================== //

// parser语法分析：函数
typedef struct Func Func;
struct Func {
    Node *body;                             // 函数体
    Var* local_vars;                        // 局部变量链表
    int stack_size;                         // 栈大小
};

// ==================================================================================== //
//                                    evoc Pub Data
// ==================================================================================== //

static char* current_input;                 // >>> 当前输入字符串

// ==================================================================================== //
//                                    evoc API: error
// ==================================================================================== //

// 编译器错误打印：打印错误信息
void evoc_err(char *fmt, ...);
// 编译器错误打印：打印字符位置
void evoc_verr_at(char *loc, char *fmt, va_list ap);
// 编译器错误打印：打印字符位置
void evoc_err_at(char *loc, char *fmt, ...);
// 编译器错误打印：打印Token
void evoc_err_tok(Token *tok, char *fmt, ...);


// ==================================================================================== //
//                                    evoc API: lexer
// ==================================================================================== //


// lexer词法分析：判断令牌是否等于op
bool token_equal(Token *tok, char *op);
// lexer词法分析：跳过令牌
Token* token_skip(Token *tok, char *op);
// lexer词法分析：识别令牌
Token* evoc_tokenize(char *p);


// ==================================================================================== //
//                                   evoc API: parser
// ==================================================================================== //

// parser语法分析：解析表达式
Func* evoc_parse(Token *tok);


// ==================================================================================== //
//                                   evoc API: codegen
// ==================================================================================== //

// codegen代码生成：生成汇编代码
void evoc_codegen(Func *prog);

#endif // EVOC_EVOC_H