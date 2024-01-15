.intel_syntax noprefix
.globl main
main:
  push rbp
  mov rbp, rsp
  sub rsp, 0
  mov rax, 3
  jmp .L.return
.L.return:
  mov rsp, rbp
  pop rbp
  ret
