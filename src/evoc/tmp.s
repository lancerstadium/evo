.intel_syntax noprefix
.globl main
main:
  push 5
  push 47
  pop rdi
  pop rax
  add rax, rdi
  push rax
  push 1
  push 9
  pop rdi
  pop rax
  imul rax, rdi
  push rax
  pop rdi
  pop rax
  sub rax, rdi
  push rax
  pop rax
  ret
