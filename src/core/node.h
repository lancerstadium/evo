

#ifndef CORE_NODE_H
#define CORE_NODE_H

typedef struct node Node;

#include "data.h"

// AST节点类型
typedef enum {
    NODE_TYPE_PROG,                     // 程序
    NODE_TYPE_MOD,                      // 模块
    NODE_TYPE_EOF,                      // 结束
    NODE_TYPE_FUNC,                     // 函数
    NODE_TYPE_EXPR,                     // 表达式
    NODE_TYPE_NUM,                      // 数字
    NODE_TYPE_IDENT,                    // 标识符
    NODE_TYPE_VAR,                      // 变量
    NODE_TYPE_BODY,                     // 代码体
    NODE_TYPE_STMT,                     // 语句
    NODE_TYPE_UNARY,                    // 单语句
    NODE_TYPE_STRUCT                    // 结构体
} NodeType;


typedef enum {
    // 表示该节点是表达式左操作数或右操作数的一部分
    NODE_FLAG_INSIDE_EXPRESSION = 0b00000001, 
} NodeFlag;


// AST节点
struct node {
    
    NodeType type;                      // AST节点类型
    int depth;                          // 节点在AST树中的深度
    Node* pnd;                          // 父节点
    int flags;                          // 节点标志

    // 存储每个节点类型的值
    union {
        char cval;
        const char* sval;
        unsigned int inum;
        unsigned long lnum;
        unsigned long long llnum;
    };

    // 存储每个节点类型单独的结构
    union {
        // 程序：prog
        struct prog {
            const char* name;                   // 程序名
            Node* main_mod;                     // 主模块
        } prog;

        // 模块：mod
        struct mod {
            const char* name;                   // 模块名
            HashMap* sym_tbl;                   // 符号表
        } mod;
        // 表达式
        struct expr {
            Node* lnd;                          // 左子节点
            Node* rnd;                          // 右子节点
            const char* op;
        } expr;

        // 单式
        struct unary {
            const char* op;
            Node* op_nd;
            union {
                struct indirection {
                    int depth;
                } indirection;
            };
        } unary;

        struct stc {
            const char* name;
            Node* body_nd;
        } stc;

        // 函数
        struct func {
            const char* name;                   // 函数名
            DataType* rtype;                    // 返回类型
            Vector* argv;                       // 传入参数表
            Node* fn_body;                      // 函数体
        } func;

        // 代码主体
        struct body {
            Vector* statements;
            size_t variable_size;
        } body;

        // 变量
        struct var {
            DataType* type;
            const char* name;
            Node* val;
            union {
                char cval;
                const char* sval;
                unsigned int inum;
                unsigned long lnum;
                unsigned long long llnum;
            } const_val;
        } var;

        // 语句
        union stmt {
            struct return_stmt {
                Node* expr_nd;
            } ret;
        } stmt;
    };

};


void node_append_size(Node* nd, size_t *var_size);
void node_read(Node* nd);
void node_swap(Node** nd1, Node** nd2);

#endif