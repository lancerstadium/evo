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
void evoc_err_at(char *loc, char *fmt, ...) {
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

## Step 7: 比较运算符
运算符优先级：
```
1. ==, !=
2. <, <=, >, >=
3. +, -
4. *, /
5. +, - (solo)
6. ()

```

表达式：
```ebnf
expr     = equality
equality = relation ("==" relation | "!=" relation)*
relation = add ("<" add | "<=" add | ">" add | ">=" add)*
add      = mul ("+" mul | "-" mul)*
mul      = unary ("*" unary | "/" unary)*
unary    = ("+" | "-")? unary | prim
prim     = num | "(" expr ")"
num      = [0-9]+

```

比较汇编，`al`位于rax低八位：
```
pop rdi
pop rax
cmp rax, rdi
sete al
movzb rax, al
```

## Step 8: C多文件





## Step 9: 标识符

```ebnf
program  = stmt*
stmt     = expr ";"
expr     = assign
assign   = equality ("=" assign)*
equality = relation ("==" relation | "!=" relation)*
relation = add ("<" add | "<=" add | ">" add | ">=" add)*
add      = mul ("+" mul | "-" mul)*
mul      = unary ("*" unary | "/" unary)*
unary    = ("+" | "-")? unary | prim
prim     = num | "(" expr ")"
num      = [0-9]+

```


## Step 10: 分号语句





## Step 11: 返回值

```
program = stmt*
stmt    = expr ";"
        | "return" expr ";"
...
```


## Step 12: 控制流

```
program = stmt*
stmt    = expr ";"
        | "if" "(" expr ")" stmt ("else" stmt)?
        | "while" "(" expr ")" stmt
        | "for" "(" expr? ";" expr? ";" expr? ")" stmt
        | "return" expr ";"
        | ...
```

```c
if(A) B

  // A: 
  pop rax
  cmp rax, 0
  je  .LendXXX
  // B:
.LendXXX:

  if (A == 0)
    goto end;
  B;
end:
```

```c
if(A) B else C

  // A: 
  pop rax
  cmp rax, 0
  je  .LendXXX
  // B:
  jmp .LendXXX
.LeleseXXX
  // C:
.LendXXX


  if (A == 0)
    goto els;
  B;
  goto end;
else:
  C;
end:

```

```c
while (A) B

.LbeginXXX:
  // A:
  pop rax
  cmp rax, 0
  je  .LendXXX
  // B:
  jmp .LbeginXXX
.LendXXX:


begin:
  if (A == 0)
    goto end;
  B;
  goto begin;
end:
```


```c
for (A; B; C) D

  // A:
.LbeginXXX:
  // B:
  pop rax
  cmp rax, 0
  je  .LendXXX
  // D:
  // C:
  jmp .LbeginXXX
.LendXXX:

  A;
begin:
  if (B == 0)
    goto end;
  D;
  C;
  goto begin;
end:

```


## Step 13: 代码块

```
program = stmt*
stmt    = expr ";"
        | "{" stmt* "}"
        | ...
```


## Step 14：无参函数

```
primary = num
        | "(" expr ")"
        | ident "(" ")"?
```