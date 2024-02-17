

#ifndef CORE_LEXER_H
#define CORE_LEXER_H

#include "token.h"
#include "scope.h"
#include "compiler.h"

#define lexer_error(...)                              \
    do {                                              \
        log_error( _bmag("[Lexer]") " " __VA_ARGS__); \
        exit(-1);                                     \
    } while(0)

#define lexer_assert(expr, ...) log_assert(expr, _bmag("[Lexer]") " " __VA_ARGS__);

typedef struct lex_process LexProcess;
typedef char (*LEX_PROCESS_NEXT_CHAR)(LexProcess* lproc);
typedef char (*LEX_PROCESS_PEEK_CHAR)(LexProcess* lproc);
typedef void (*LEX_PROCESS_PUSH_CHAR)(LexProcess* lproc, char c);

// 词法分析结果的状态
typedef enum {
    LEXER_ANALYSIS_OK,                  // 词法分析成功
    LEXER_ANALYSIS_ERROR                // 词法分析失败
} LexerAnalysisResult;

// lexer进程
struct lex_process {

    Pos pos;                            // 当前字符位置
    Pos last_pos;                       // 上一个字符的位置
    Token tmp_tok;                      // 用于存储上一个 token
    Vector* token_vec;                  // 用于储存生成的 token
    CompileProcess* compile_proc;       // 指向 compile_process 的指针

    bool paren_unclose;                 // 上个括号未闭合
    Vector* parens_vec;                 // 括号堆栈 token
    int cur_expr_depth;                 // 记录当前处于几层括号`()`夹层内
    int cur_list_depth;                 // 记录当前处于几层括号`[]`夹层内
    int cur_scope_depth;                // 记录当前处于几层括号`{}`夹层内

    struct {
        bool macro_def;                 // 是否在宏定义
        HashMap* macro_sym_tbl;         // 存储宏的 hashmap
    } pre;

    LEX_PROCESS_NEXT_CHAR next_char;    // 读取下一字符，进一位
    LEX_PROCESS_PEEK_CHAR peek_char;    // 读取下一字符，留在原地
    LEX_PROCESS_PUSH_CHAR push_char;    // 将字符 c 替换当前字符，并退一位

    void* priv;                         // 用于储存 lexer 都不懂的东西
};

// lexer.c
int lex(LexProcess* lex_proc);

// lex_proc.c
LexProcess* lex_process_create(CompileProcess* cproc, void* priv);
void lex_process_free(LexProcess* lproc);
Vector* lex_process_tokens(LexProcess* lproc);
void* lex_process_private(LexProcess* lproc);
Token* lex_process_next_token(LexProcess* lproc);

#endif // CORE_LEXER_H