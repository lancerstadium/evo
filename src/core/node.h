

#ifndef CORE_NODE_H
#define CORE_NODE_H

typedef struct node Node;

#include "data.h"

// AST节点类型
typedef enum {
    NODE_TYPE_EXPR,                     // 表达式
    NODE_TYPE_NUM,                      // 数字
    NODE_TYPE_IDENT,                    // 标识符
    NODE_TYPE_VAR,                      // 变量
    NODE_TYPE_FUNC,                     // 函数
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
        // 表达式
        struct expr {
            Node* left;
            Node* right;
            const char* op;
        } expr;

        // 括号表达式：(expr)
        struct parenthesis {
            Node* expr_nd;
        } parenthesis;

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
            DataType* rtype;
            const char* name;
            Vector* argv;
            Node* body_nd;
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