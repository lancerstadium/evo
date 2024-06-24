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
