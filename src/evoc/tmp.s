.intel_syntax noprefix
.globl main
main:
  push 1
  push 2
  pop rdi
  pop rax
  cmp rax, rdi
  setge al
  movzb rax, al
  push rax
  pop rax
  ret
