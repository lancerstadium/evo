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