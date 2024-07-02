
#include <evo/evo.h>

/**
 * @file def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief x86 isa defination
 * @date 2024-07-01
 * 
 * @copyright Copyright (c) 2024
 * 
 * @note
 * - [x86 Insn](http://ref.x86asm.net/coder32.html)
 * - [x86 Disasm](https://defuse.ca/online-x86-assembler.htm)
 * 
 * 
 *  ┌────────┬───────────┬───────────┬──────────┬─────────┐
 *  │ Encode │ r32[31:0] │ r16[15:0] │ r8[15:8] │ r8[7:0] │
 *  ├────────┼───────────┼───────────┼──────────┼─────────┤
 *  │  000   │    eax    │    ax     │    ah    │   al    │  
 *  │  001   │    ecx    │    cx     │    ch    │   cl    │
 *  │  010   │    edx    │    dx     │    dh    │   dl    │
 *  │  011   │    ebx    │    bx     │    bh    │   bl    │
 *  │  100   │    esp    │    sp     │    --    │   --    │
 *  │  101   │    ebp    │    bp     │    --    │   --    │
 *  │  110   │    esi    │    si     │    --    │   --    │
 *  │  111   │    edi    │    di     │    --    │   --    │
 *  └────────┴───────────┴───────────┴──────────┴─────────┘
 * 
 */






#ifndef _ISA_X86_DEF_H_
#define _ISA_X86_DEF_H_

#include <evo/evo.h>


#define X86_EL(I)    X86_##I

// ==================================================================================== //
//                                    x86: Reg                                      
// ==================================================================================== //

RegID_def(X86,
    /* GPR 32-bit */
    REP8(X86_EL, EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI),
    /* GPR 16-bit */
    REP8(X86_EL, AX, CX, DX, BX, SP, BP, SI, DI),
    /* GPR 8-bit */
    REP8(X86_EL, AL, CL, DL, BL, AH, CH, DH, BH),
);

RegDef_def(X86,
    [X86_EAX]   = { .id = X86_EAX           , .name = "eax" , .alias = "eax"   , .map = {31, 0} },
    [X86_ECX]   = { .id = X86_ECX           , .name = "ecx" , .alias = "ecx"   , .map = {31, 0} },
    [X86_EDX]   = { .id = X86_EDX           , .name = "edx" , .alias = "edx"   , .map = {31, 0} },
    [X86_EBX]   = { .id = X86_EBX           , .name = "ebx" , .alias = "ebx"   , .map = {31, 0} },
    [X86_ESP]   = { .id = X86_ESP           , .name = "esp" , .alias = "esp"   , .map = {31, 0} },
    [X86_EBP]   = { .id = X86_EBP           , .name = "ebp" , .alias = "ebp"   , .map = {31, 0} },
    [X86_ESI]   = { .id = X86_ESI           , .name = "esi" , .alias = "esi"   , .map = {31, 0} },
    [X86_EDI]   = { .id = X86_EDI           , .name = "edi" , .alias = "edi"   , .map = {31, 0} },

    [X86_AX]    = { .id = X86_EAX           , .name = "ax"   , .alias = "eax"   , .map = {15, 0} },
    [X86_CX]    = { .id = X86_ECX           , .name = "cx"   , .alias = "ecx"   , .map = {15, 0} },
    [X86_DX]    = { .id = X86_EDX           , .name = "dx"   , .alias = "edx"   , .map = {15, 0} },
    [X86_BX]    = { .id = X86_EBX           , .name = "bx"   , .alias = "ebx"   , .map = {15, 0} },
    [X86_SP]    = { .id = X86_ESP           , .name = "sp"   , .alias = "esp"   , .map = {15, 0} },
    [X86_BP]    = { .id = X86_EBP           , .name = "bp"   , .alias = "ebp"   , .map = {15, 0} },
    [X86_SI]    = { .id = X86_ESI           , .name = "si"   , .alias = "esi"   , .map = {15, 0} },
    [X86_DI]    = { .id = X86_EDI           , .name = "di"   , .alias = "edi"   , .map = {15, 0} },

    [X86_AL]    = { .id = X86_EAX           , .name = "al"   , .alias = "eax"   , .map = { 7, 0} },
    [X86_CL]    = { .id = X86_ECX           , .name = "cl"   , .alias = "ecx"   , .map = { 7, 0} },
    [X86_DL]    = { .id = X86_EDX           , .name = "dl"   , .alias = "edx"   , .map = { 7, 0} },
    [X86_BL]    = { .id = X86_EBX           , .name = "bl"   , .alias = "ebx"   , .map = { 7, 0} },
    [X86_AH]    = { .id = X86_EAX           , .name = "ah"   , .alias = "eax"   , .map = {15, 8} },
    [X86_CH]    = { .id = X86_ECX           , .name = "ch"   , .alias = "ecx"   , .map = {15, 8} },
    [X86_DH]    = { .id = X86_EDX           , .name = "dh"   , .alias = "edx"   , .map = {15, 8} },
    [X86_BH]    = { .id = X86_EBX           , .name = "bh"   , .alias = "ebx"   , .map = {15, 8} },

);

// ==================================================================================== //
//                                    x86: Insn                                      
// ==================================================================================== //


InsnID_def(X86, 
    /* X86: None    */
    REP1(X86_EL, INV),
    REP2(X86_EL, RET, LEAVE),
    REP2(X86_EL, CWTL, CLTD),
    REP1(X86_EL, MOVS),
    REP2(X86_EL, REP_MOVS, REP_STOS),
    REP3(X86_EL, PUSHF, PUSHA, POPA),
    REP3(X86_EL, CLC, STC, CLD),
    REP1(X86_EL, IRET),
    REP2(X86_EL, CPUID, RDTSC),
    /* X86: Unary   */
    REP6(X86_EL, CALL, JCC, JMP, SETCC, CALL_E, JMP_E),
    REP2(X86_EL, PUSH, POP),
    REP4(X86_EL, INC, DEC, NEG, NOT),
    REP4(X86_EL, MUL, IMUL1, DIV, IDIV),
    REP4(X86_EL, LGDT, LIDT, LTR, INT),
    /* X86: Binary  */
    REP6(X86_EL, MOV, ADD, SUB, ADC, SBB, CMP),
    REP4(X86_EL, AND, OR, TEST, XOR),
    REP5(X86_EL, SHL, SHR, SAR, ROL, ROR),
    REP5(X86_EL, LEA, MOVZB, MOVZW, MOVSB, MOVSW),
    REP1(X86_EL, IMUL2),
    REP4(X86_EL, BSR, BT, XCHG, CMPXCHG),
    /* X86: Ternary */
    REP3(X86_EL, IMUL3, SHLD, SHRD),
);

#define Tys_x86op1()    Tys_new(Ty_I(0, {7, 0}))
#define Tys_x86op2()    Tys_new(Ty_I(0, {7, 0}), Ty_I(0, {15, 8}))

#define Tys_x86r()      Tys_new(Ty_r(0, {15, 8}))


InsnDef_def(X86,
    [X86_ADC]   = { .id = X86_ADC   , .mnem = "adc" , .bc = Val_u8(0x01)    , .tc = Tys_x86op1()    , .tr = Tys_x86r()  },
);



Insn_def(X86

,

);

// ==================================================================================== //
//                                    x86: Block                                      
// ==================================================================================== //

Block_def(X86);


#endif // _ISA_X86_DEF_H_