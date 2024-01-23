
#ifndef CORE_PARSER_H
#define CORE_PARSER_H

#include "lexer.h"
#include "node.h"

#define parser_error(...) \
    do { \
    log_error( _bmag("[Parser]") " " __VA_ARGS__); \
    exit(-1); \
    } while(0)

// 类型声明
typedef struct parse_process ParseProcess;
typedef Token* (*PARSE_PROCESS_NEXT_TK)(ParseProcess* pproc);
typedef Token* (*PARSE_PROCESS_PEEK_TK)(ParseProcess* pproc);
typedef Token* (*PARSE_PROCESS_EXCP_TK)(ParseProcess* pproc, NodeType type);
typedef Node* (*PARSE_PROCESS_PEEK_ND)(ParseProcess* pproc);
typedef Node* (*PARSE_PROCESS_POP_ND)(ParseProcess* pproc);
typedef void (*PARSE_PROCESS_PUSH_ND)(ParseProcess* pproc, Node* node);
typedef Node* (*PARSE_PROCESS_CREATE_ND)(ParseProcess* pproc, Node* _node);


// 语法分析结果的状态
typedef enum {
    PARSER_ANALYSIS_OK,                 // 语法分析成功
    PARSER_ANALYSIS_ERROR,              // 语法分析失败                
} ParserAnalysisResult;

// parser进程
struct parse_process {

    Vector* node_tree_vec;              // 包含指向树根的指针
    Vector* node_vec;                   // 用于存储解析所有节点：可以被弹出以形成其他更大的节点，例如表达式
    LexProcess* lex_proc;               // 指向 lex_process 的指针

    PARSE_PROCESS_NEXT_TK next_token;   // 移到下一个 token
    PARSE_PROCESS_PEEK_TK peek_token;   // 查看下一个 token
    PARSE_PROCESS_EXCP_TK excp_token;   // 期望下一个 token 是什么

    PARSE_PROCESS_PEEK_ND peek_node;
    PARSE_PROCESS_POP_ND  pop_node;
    PARSE_PROCESS_PUSH_ND push_node;
    PARSE_PROCESS_CREATE_ND create_node;

};


// parser.c
int parse(ParseProcess *pproc);

// parse_proc.c
ParseProcess* parse_process_create(LexProcess* lproc);
void parse_process_free(ParseProcess* pproc);
int parse_process_next(ParseProcess* pproc);

#endif 