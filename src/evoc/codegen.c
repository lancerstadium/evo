
// ==================================================================================== //
//                                     Evoc: codegen
// ==================================================================================== //

#include "evoc.h"

// ==================================================================================== //
//                                   Pri Data: codegen
// ==================================================================================== //

static int depth;                                       // 栈深度

// ==================================================================================== //
//                                   Pri API: codegen
// ==================================================================================== //

static void push(void) {
    printf("  push rax\n");                             // 入栈：变量
    depth++;                                            // 深度加一
}

static void pop(char *reg) {
    printf("  pop %s\n", reg);                          // 出栈：变量
    depth--;                                            // 深度减一
}
// 生成表达式
static void gen_expr(Node *node) {
    switch(node->type) {
        case ND_NUM:                                    // 如果为数字
            printf("  mov rax, %d\n", node->val);       // rax = 数字
            return;
        case ND_NEG:                                    // 如果为unary -
            gen_expr(node->lhs);                        // 生成表达式
            printf("  neg rax\n");                      // rax = -rax
            return;
    }

    gen_expr(node->rhs);                                // 生成右表达式
    push();                                             // 入栈：rax
    gen_expr(node->lhs);                                // 生成左表达式
    pop("rdi");                                         // 出栈：rdi

    switch(node->type) {
        case ND_ADD:                                    // 如果为`+`
            printf("  add rax, rdi\n"); return;         // rax += rdi
        case ND_SUB:                                    // 如果为`-`
            printf("  sub rax, rdi\n"); return;         // rax -= rdi
        case ND_MUL:                                    // 如果为`*`
            printf("  imul rax, rdi\n"); return;        // rax *= rdi
        case ND_DIV:                                    // 如果为`/`
            printf("  cqo\n");                          // cqo：rax = rdx:rax
            printf("  idiv rdi\n"); return;             // rax /= rdi ... rdx
        case ND_EQU:                                    // 如果为`==`
            printf("  cmp rax, rdi\n");                 // rax == rdi
            printf("  sete al\n");                      // al = rax == rdi
            printf("  movzb rax, al\n"); return;        // rax = al
        case ND_NEQ:                                    // 如果为`!=`
            printf("  cmp rax, rdi\n");                 // rax != rdi
            printf("  setne al\n");                     // al = rax != rdi
            printf("  movzb rax, al\n"); return;        // rax = al
        case ND_LSS:                                    // 如果为`<`
            printf("  cmp rax, rdi\n");                 // rax < rdi
            printf("  setl al\n");                      // al = rax < rdi
            printf("  movzb rax, al\n"); return;        // rax = al
        case ND_GTR:                                    // 如果为`>`
            printf("  cmp rax, rdi\n");                 // rax > rdi
            printf("  setg al\n");                      // al = rax > rdi
            printf("  movzb rax, al\n"); return;        // rax = al
        case ND_LEQ:                                    // 如果为`<=`
            printf("  cmp rax, rdi\n");                 // rax <= rdi
            printf("  setle al\n");                     // al = rax <= rdi
            printf("  movzb rax, al\n"); return;        // rax = al
        case ND_GEQ:                                    // 如果为`>=`
            printf("  cmp rax, rdi\n");                 // rax >= rdi
            printf("  setge al\n");                     // al = rax >= rdi
            printf("  movzb rax, al\n"); return;        // rax = al
    }
    log_error("invalid expression");
}
// 生成语句
static void gen_stmt(Node *node) {
    if (node->type == ND_EXPR_STMT) {
        gen_expr(node->lhs);
        return;
    }
    log_error("invalid statement");
}



// ==================================================================================== //
//                                   Pub API: codegen
// ==================================================================================== //

// codegen：生成汇编代码
// void codegen(Node *node) {
//     if(node->type == ND_RETURN) {
//         codegen(node->lhs);
//         printf("  pop rax\n");                      // 出栈：返回值
//         printf("  mov rsp, rbp\n");                 // rsp = rbp
//         printf("  pop rbp\n");                      // 出栈：rbp
//         printf("  ret\n");                          // 返回
//     }
//     switch(node->type) {
//         case ND_NUM:                                // 如果为数字
//             printf("  push %d\n", node->val);       // 入栈：数字
//             break;
//         case ND_VAR:                                // 如果为变量
//             gen_val(node);                          // 生成变量节点
//             printf("  push rax\n");                 // 入栈：变量
//             printf("  mov rax, [rax]\n");           // 变量取值：rax = [rax]
//             printf("  push rax\n");                 // 入栈：变量值
//             break;
//         case ND_ASSIGN:                             // 如果为赋值
//             gen_val(node->lhs);                     // 生成左变量节点
//             gen_val(node->rhs);                     // 生成右变量节点
//             printf("  pop rdi\n");                  // 出栈：右操作数
//             printf("  pop rax\n");                  // 出栈：左操作数
//             printf("  mov [rax], rdi\n");           // 变量赋值
//             printf("  push rdi\n");                 // 入栈：变量值
//             break;
//     }
//     codegen(node->lhs);
//     codegen(node->rhs);
//     printf("  pop rdi\n");
//     printf("  pop rax\n");
//     switch(node->type) {
//         case ND_ADD:                                // 如果为`+`
//             printf("  add rax, rdi\n"); break;      // rax += rdi
//         case ND_SUB:                                // 如果为`-`
//             printf("  sub rax, rdi\n"); break;      // rax -= rdi
//         case ND_MUL:                                // 如果为`*`
//             printf("  imul rax, rdi\n"); break;     // rax *= rdi
//         case ND_DIV:                                // 如果为`/`
//             printf("  cqo\n");                      // cqo ：rax = rdx:rax
//             printf("  idiv rdi\n"); break;          // rax /= rdi ... rdx
//         case ND_EQU:                                // 如果为`==`
//             printf("  cmp rax, rdi\n");             // rax == rdi
//             printf("  sete al\n");                  // al = rax == rdi
//             printf("  movzb rax, al\n"); break;     // rax = al
//         case ND_NEQ:                                // 如果为`!=`
//             printf("  cmp rax, rdi\n");             // rax != rdi
//             printf("  setne al\n");                 // al = rax != rdi
//             printf("  movzb rax, al\n"); break;     // rax = al
//         case ND_LSS:                                // 如果为`<`
//             printf("  cmp rax, rdi\n");             // rax < rdi
//             printf("  setl al\n");                  // al = rax < rdi
//             printf("  movzb rax, al\n"); break;     // rax = al
//         case ND_GTR:                                // 如果为`>`
//             printf("  cmp rax, rdi\n");             // rax > rdi
//             printf("  setg al\n");                  // al = rax > rdi
//             printf("  movzb rax, al\n"); break;     // rax = al
//         case ND_LEQ:                                // 如果为`<=`
//             printf("  cmp rax, rdi\n");             // rax <= rdi
//             printf("  setle al\n");                 // al = rax <= rdi
//             printf("  movzb rax, al\n"); break;     // rax = al
//         case ND_GEQ:                                // 如果为`>=`
//             printf("  cmp rax, rdi\n");             // rax >= rdi
//             printf("  setge al\n");                 // al = rax >= rdi
//             printf("  movzb rax, al\n"); break;     // rax = al
//     }

//     printf("  push rax\n");                         // 入栈：结果
// }
void evoc_codegen(Node *node) {
    printf(".intel_syntax noprefix\n");             // 设置汇编语法格式：INTEL，无前缀
    printf(".globl main\n");                        // 定义全局变量：main
    printf("main:\n");                              // 定义main函数
    for (Node *n = node; n; n = n->next) {
        gen_stmt(n);
        // log_assert(depth == 0);                     // 检查栈深度
    }
    printf("  ret\n");                              // 返回
}
