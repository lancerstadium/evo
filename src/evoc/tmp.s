.intel_syntax noprefix
.globl main
main:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  lea rax, [rbp - 8]
  push rax
  mov rax, 4
  pop rdi
  mov [rax], rdi
  lea rax, [rbp - 8]
  mov rax, [rax]
  jmp .L.return
.L.return:
  mov rsp, rbp
  pop rbp
  ret
