.intel_syntax noprefix
.globl main
main:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  lea rax, [rbp - 8]
  push rax
  mov rax, 0
  pop rdi
  mov [rax], rdi
.L.begin.1:
  mov rax, 10
  push rax
  lea rax, [rbp - 8]
  mov rax, [rax]
  pop rdi
  cmp rax, rdi
  setl al
  movzb rax, al
  cmp rax, 0
  je .L.end.1
  lea rax, [rbp - 8]
  push rax
  mov rax, 1
  push rax
  lea rax, [rbp - 8]
  mov rax, [rax]
  pop rdi
  add rax, rdi
  pop rdi
  mov [rax], rdi
  jmp .L.begin.1
.L.end.1:
  lea rax, [rbp - 8]
  mov rax, [rax]
  jmp .L.return
.L.return:
  mov rsp, rbp
  pop rbp
  ret
