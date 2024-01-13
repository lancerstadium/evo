


#ifndef EVOC_PARSER_H
#define EVOC_PARSER_H

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include "lexer.h"

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
    ND_EQU,                 // ==
    ND_NEQ,                 // !=
    ND_LSS,                 // <
    ND_GTR,                 // >
    ND_LEQ,                 // <=
    ND_GEQ,                 // >=
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
    Node *node = (Node*)calloc(1, sizeof(Node));
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
Node* equality();
Node* relation();
Node* add();
Node* mul();
Node* unary();
Node* prim();

// parser语法分析：表达式 `expr = equality`
Node* expr() {
    return equality();
}
// parser语法分析：表达式 `equality = relation ("==" relation | "!=" relation)*`
Node* equality() {
    Node *node = relation();
    while(true) {
        if(consume("==")) {
            node = node_new_binary(ND_EQU, node, relation());
        }else if(consume("!=")) {
            node = node_new_binary(ND_NEQ, node, relation());
        }else {
            return node;
        }
    }
}
// parser语法分析：表达式 `relation = add ("<" add | ">" add | "<=" add | ">=" add)*`
Node* relation() {
    Node *node = add();
    while(true) {
        if(consume("<")) {
            node = node_new_binary(ND_LSS, node, add());
        }else if(consume(">")) {
            node = node_new_binary(ND_GTR, node, add());
        }else if(consume("<=")) {
            node = node_new_binary(ND_LEQ, node, add());
        }else if(consume(">=")) {
            node = node_new_binary(ND_GEQ, node, add());
        }else {
            return node;
        }
    }
}
// parser语法分析：表达式 `add = mul ("+" mul | "-" mul)*`
Node* add() {
    Node *node = mul();
    while(true) {
        if(consume("+")) {
            node = node_new_binary(ND_ADD, node, mul());
        }else if(consume("-")) {
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
        if(consume("*")) {
            node = node_new_binary(ND_MUL, node, unary());
        }else if(consume("/")) {
            node = node_new_binary(ND_DIV, node, unary());
        }else {
            return node;
        }
    }
}
// parser语法分析：表达式 `unary = ("+" | "-")? unary | prim`
Node* unary() {
    if(consume("+")) {
        return unary();
    }else if(consume("-")) {
        return node_new_binary(ND_SUB, node_new_num(0), unary());
    }
    return prim();
}

// parser语法分析：表达式 `prim = num | "(" expr ")"`
Node* prim() {
    if(consume("(")) {
        Node *node = expr();
        expect(")");
        return node;
    }
    return node_new_num(expect_number());
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
            printf("  idiv rdi\n"); break;          // rax /= rdi ... rdx
        case ND_EQU:                                // 如果为`==`
            printf("  cmp rax, rdi\n");             // rax == rdi
            printf("  sete al\n");                  // al = rax == rdi
            printf("  movzb rax, al\n"); break;     // rax = al
        case ND_NEQ:                                // 如果为`!=`
            printf("  cmp rax, rdi\n");             // rax != rdi
            printf("  setne al\n");                 // al = rax != rdi
            printf("  movzb rax, al\n"); break;     // rax = al
        case ND_LSS:                                // 如果为`<`
            printf("  cmp rax, rdi\n");             // rax < rdi
            printf("  setl al\n");                  // al = rax < rdi
            printf("  movzb rax, al\n"); break;     // rax = al
        case ND_GTR:                                // 如果为`>`
            printf("  cmp rax, rdi\n");             // rax > rdi
            printf("  setg al\n");                  // al = rax > rdi
            printf("  movzb rax, al\n"); break;     // rax = al
        case ND_LEQ:                                // 如果为`<=`
            printf("  cmp rax, rdi\n");             // rax <= rdi
            printf("  setle al\n");                 // al = rax <= rdi
            printf("  movzb rax, al\n"); break;     // rax = al
        case ND_GEQ:                                // 如果为`>=`
            printf("  cmp rax, rdi\n");             // rax >= rdi
            printf("  setge al\n");                 // al = rax >= rdi
            printf("  movzb rax, al\n"); break;     // rax = al
    }

    printf("  push rax\n");                         // 入栈：结果
}

#endif // EVOC_PARSER_H