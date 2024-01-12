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
