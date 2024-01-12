#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "usage: %s <int>\n", argv[0]);
        return 1;
    }

    char *p = argv[1];

    printf(".intel_syntax noprefix\n");                     // 设置汇编语法格式：INTEL，无前缀
    printf(".globl main\n");                                // 定义全局变量：main
    printf("main:\n");                                      // 定义main函数
    printf("  mov rax, %ld\n", strtol(p, &p, 10));          // mov rax, argv[1]

    while(*p) {
        if(*p == '+') {                                     // 加号：+
            p++;
            printf("  add rax, %ld\n", strtol(p, &p, 10));  // add rax, argv[3]
            continue;
        }
        if(*p == '-') {                                     // 减号：-
            p++;
            printf("  sub rax, %ld\n", strtol(p, &p, 10));  // sub rax, argv[3]
            continue;
        }

        fprintf(stderr, "unknown operator: %c\n", *p);      // 未知运算符
        return 1;
    }
    printf("  ret\n");                                      // 返回
    return 0;
}