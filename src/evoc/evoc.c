

// ==================================================================================== //
//                                       Include
// ==================================================================================== //

#include "evoc.h"

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