.intel_syntax noprefix
.globl main
main:
  push rbp
  mov rbp, rsp
  sub rsp, 0
  mov rax, 2
  push rax
  mov rax, 1
  pop rdi
  cmp rax, rdi
  setge al
  movzb rax, al
  jmp .L.return
.L.return:
  mov rsp, rbp
  pop rbp
  ret
