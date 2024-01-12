# Assmble generation of evo

## Intel and AT&T

- 默认情况下，gcc 和 objdump 以 AT&T 表示法输出程序集。
- 在AT&T表示法中，结果寄存器是第二个参数。 因此，在双参数指令中，参数以相反的顺序编写。 用 % 前缀写寄存器名称， %rax 依此类推。 数字 $ 以前缀 $42 
- 此外，在引用内存时， [] () 请使用 instead to 以唯一表示法编写表达式。 以下是一些对比示例：
```
mov rbp, rsp   // Intel
mov %rsp, %rbp // AT&T
mov rax, 8     // Intel
mov $8, %rax   // AT&T
mov [rbp + rcx * 4 - 8], rax // Intel
mov %rax, -8(rbp, rcx, 4)    // AT&T

```

- 在此编译器中，我决定使用英特尔符号来提高可读性。 由于英特尔的指令集手册使用英特尔符号，因此它的优点是能够按照代码中的原样编写手册的描述。 AT&T和Intel符号的表现力是相同的。 无论使用哪种表示法，生成的机器指令序列都是相同的。