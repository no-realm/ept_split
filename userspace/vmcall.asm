section .text

global _platform_vmcall
_platform_vmcall:

    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    mov rdi, rcx

    mov rax, [rdi + 00h]
    mov rdx, [rdi + 08h]
    mov rcx, [rdi + 10h]
    mov rbx, [rdi + 18h]
    mov rsi, [rdi + 20h]
    mov r8,  [rdi + 28h]
    mov r9,  [rdi + 30h]
    mov r10, [rdi + 38h]
    mov r11, [rdi + 40h]
    mov r12, [rdi + 48h]
    mov r13, [rdi + 50h]
    mov r14, [rdi + 58h]
    mov r15, [rdi + 60h]

    vmcall

    mov [rdi + 60h], r15
    mov [rdi + 58h], r14
    mov [rdi + 50h], r13
    mov [rdi + 48h], r12
    mov [rdi + 40h], r11
    mov [rdi + 38h], r10
    mov [rdi + 30h], r9
    mov [rdi + 28h], r8
    mov [rdi + 20h], rsi
    mov [rdi + 18h], rbx
    mov [rdi + 10h], rcx
    mov [rdi + 08h], rdx

    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbx

    ret
