# Language design of evo


## Step 1: 整数输出

```
.intel_syntax noprefix
.globl main
main:
        mov rax, 5
        add rax, 20
        sub rax, 4
        ret
```

## Step 2: 加法与减法

```
.intel_syntax noprefix
.globl main
main:
  mov rax, 5
  add rax, 20
  sub rax, 4
  ret
```

## Step 3: 识别标记
- 标记之间的空格仅用于分隔标记，而不是组成单词的部分。 
- 因此，在将字符串拆分为标记列时，去掉空格是很自然的。 将字符串拆分为标记字符串称为“标记化”（Tokenize）。
```c
// lexer词法分析：令牌结构体
typedef struct Token Token;
struct Token {
    TokenType type;         // 令牌类型
    Token *next;            // 下一个令牌
    int val;                // 令牌值
    char *str;              // 令牌字符串
};

```



## Step 4: 改进错误信息
```c
// 编译器错误打印
void evoc_err(char *loc, char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);                          // 初始化ap
    int pos = loc - user_input;                 // 错误位置
    fprintf(stderr, "%s\n", user_input);        // 打印用户输入字符串
    fprintf(stderr, "%*s", pos, " ");           // 打印错误位置
    fprintf(stderr, "^ ");                      // 打印错误位置
    vfprintf(stderr, fmt, ap);                  // 打印错误信息
    va_end(ap);                                 // 释放ap
    fprintf(stderr,"\n");                       // 换行
}
```

## Step 5: EBNF 表达式生成规则

```ebnf
expr = mul ("+" mul | "-" mul)*
mul  = prim ("*" prim | "/" prim)*
prim = num | "(" expr ")"
num  = [0-9]+
```

## Step 6: 一元运算符
```ebnf
expr = mul ("+" mul | "-" mul)*
mul  = unary ("*" unary | "/" unary)*
unary = ("+" | "-")? prim
prim = num | "(" expr ")"
num  = [0-9]+
```