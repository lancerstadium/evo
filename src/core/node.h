

#ifndef CORE_NODE_H
#define CORE_NODE_H

typedef struct node Node;

#include "data.h"

// AST节点类型
typedef enum {
    NODE_TYPE_EXPRESSION,               // 表达式
    NODE_TYPE_NUMBER,                   // 数字
    NODE_TYPE_IDENTIFIER,               // 标识符
    NODE_TYPE_VARIABLE,                 // 变量
    NODE_TYPE_FUNCTION                  // 函数
} NodeType;

// AST节点
struct node {
    NodeType type;                      // AST节点类型

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
        // 函数
        struct func {
            DataType* type;
        } func;
        // 变量
        struct var {
            DataType* type;
            const char* name;
            Node* val;
        } var;
    };
};


void node_read(Node* nd);


#endif