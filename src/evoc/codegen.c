
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

static int count(void) {
    static int i = 1;
    return i++;
}

// 将`n`向最接近的`align`的倍数取整。
// 例如，align_to(5, 8)返回8，align_to(11, 8)返回16。
static int align_to(int n, int align) {
    return (n + align - 1) / align * align;
}
// 对齐本地变量的偏移量
static void assign_local_vars_offset(Func *prog) {
    int offset = 0;
    for(Var *var = prog->local_vars; var; var = var->next) {
        offset += 8;
        var->offset = offset;
    }
    prog->stack_size = align_to(offset, 16);            // 栈大小为16倍数
}

// 生成地址：计算给定节点的绝对地址。
// 如果给定节点不在内存中，则属于错误。
static void gen_addr(Node *node) {
    if(node->type == ND_VAR) {                          // 如果为变量
        // rax = rbp - offset
        printf("  lea rax, [rbp - %d]\n", node->var->offset);      
        return;
    }
    evoc_err("not an lvalue");
}

// 生成表达式
static void gen_expr(Node *node) {
    // 处理符号和数字
    switch(node->type) {
        case ND_NUM:                                    // 如果为数字
            printf("  mov rax, %d\n", node->val);       // rax = 数字
            return;
        case ND_NEG:                                    // 如果为unary -
            gen_expr(node->lhs);                        // 生成表达式
            printf("  neg rax\n"); return;              // rax = -rax
        case ND_VAR:                                    // 如果为变量
            gen_addr(node);                             // 生成地址
            printf("  mov rax, [rax]\n"); return;       // rax = [rax]
        case ND_ASSIGN:                                 // 如果为赋值
            gen_addr(node->lhs);                        // 生成左表达式地址
            push();                                     // 入栈：rax
            gen_expr(node->rhs);                        // 生成右表达式
            pop("rdi");                                 // 出栈：rdi
            printf("  mov [rax], rdi\n"); return;       // [rax] = rdi
    }
    gen_expr(node->rhs);                                // 生成右表达式
    push();                                             // 入栈：rax
    gen_expr(node->lhs);                                // 生成左表达式
    pop("rdi");                                         // 出栈：rdi
    // 处理运算
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
    switch(node->type) {
        case ND_IF: {                                   // 如果为`if`
            int c = count();                            // 计数
            printf("  cmp rax, 0\n");                   // rax == 0
            printf("  je .L.else.%d\n", c);             // if rax == 0 goto .L.else
            gen_stmt(node->then);                       // 生成then
            printf("  jmp .L.end.%d\n", c);             // goto .L.end
            printf(".L.else.%d:\n", c);                 // .L.else
            if(node->els) {
                gen_stmt(node->els);                    // 生成else
            }
            printf(".L.end.%d:\n", c); return;          // .L.end
        }
        case ND_LOOP: {                                 // 如果为`for`或`while`
            int c = count();
            if(node->init) {                            // for(init; cond; inc)
                gen_stmt(node->init);
            }
            printf(".L.begin.%d:\n", c);                // .L.begin
            if(node->cond) {
                gen_expr(node->cond);                   // condition
                printf("  cmp rax, 0\n");               // rax == 0
                printf("  je .L.end.%d\n", c);          // goto .L.end
            }
            gen_stmt(node->then);                       // then
            if(node->inc) {                             // inc
                gen_expr(node->inc);
            }
            printf("  jmp .L.begin.%d\n", c);           // goto .L.begin
            printf(".L.end.%d:\n", c);                  // .L.end
            return;
        }
        case ND_BLOCK: {                                // 如果为代码块
            for(Node *n = node->body; n; n = n->next) { // 遍历代码块
                gen_stmt(n);
            }
            return;
        }
        case ND_RETURN: {                               // 如果为return
            gen_expr(node->lhs);                        // 生成表达式
            printf("  jmp .L.return\n"); return;        // return
        }
        case ND_EXPR_STMT: {                            // 如果为表达式语句
            gen_expr(node->lhs);                        // 生成表达式
            return;
        }
    }
    log_error("invalid statement: %d", node->type);
}



// ==================================================================================== //
//                                   Pub API: codegen
// ==================================================================================== //

// codegen：生成汇编代码
void evoc_codegen(Func *prog) {
    assign_local_vars_offset(prog);                 // 对齐本地变量的偏移量

    printf(".intel_syntax noprefix\n");             // 设置汇编语法格式：INTEL，无前缀
    printf(".globl main\n");                        // 定义全局变量：main
    printf("main:\n");                              // 定义main函数

    // 初始化栈：大小为208
    printf("  push rbp\n");                         // 入栈：rbp
    printf("  mov rbp, rsp\n");                     // rbp = rsp
    printf("  sub rsp, %d\n", prog->stack_size);    // rsp -= 栈大小
    
    gen_stmt(prog->body);                           // 生成代码
    // log_assert(depth == 0);                         // 检查栈深度

    printf(".L.return:\n");                         // return
    printf("  mov rsp, rbp\n");                     // rsp = rbp
    printf("  pop rbp\n");                          // 出栈：rbp
    printf("  ret\n");                              // 返回
}
