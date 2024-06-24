# ASMORG

## Simplest assembler ever

ASMORG is a simple assembler which turns AT&T Assembly code into a 32-bit statically linked executable ELF file.

## How to build

```bash
gcc -o asmorg asmorg.c -ansi -pedantic -Wall -Werror
```

## How to assemble a file

```bash
./asmorg <input_file> [<output_file>]
```

## Example

Let's write a hello world program for Linux, you can find this file in ```tests/hello.s```.

```asm
main:
    pushl $104
    call putc
    popl %eax
    pushl $101
    call putc
    popl %eax
    pushl $108
    call putc
    popl %eax
    pushl $108
    call putc
    popl %eax
    pushl $111
    call putc
    popl %eax

    movl $1,%eax
    movl $0,%ebx
    syscall
    ret

putc:
    pushl %ebp
    movl %esp,%ebp
    movl $4,%eax
    movl $1,%ebx
    movl %ebp,%ecx
    addl $8,%ecx
    movl $1,%edx
    syscall
    leave
    ret
```

Now assemble it.

```bash
./asmorg tests/hello.s hello
```

Give the ```hello``` file execute permissions.

```bash
chmod +x hello
```

Now you can run it.

```bash
./hello
```

This is the output.

```
hello
```
