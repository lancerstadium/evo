.intel_syntax noprefix
.globl main
main:
  mov rax, 2
  push rax
  mov rax, 1
  pop rdi
  cmp rax, rdi
  setge al
  movzb rax, al
  ret
