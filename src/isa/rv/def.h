/**
 * @file isa/rv/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief riscv isa defination
 * @date 2024-06-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */



#ifndef _ISA_RV_DEF_H_
#define _ISA_RV_DEF_H_

#include <evo/evo.h>


#define RV_EL(I)    RV_##I


// ==================================================================================== //
//                                    rv: Reg                                      
// ==================================================================================== //

RegID_def(RV,
    /* GPR */
    REP8(RV_EL, R0, R1, R2, R3, R4, R5, R6, R7),
    REP8(RV_EL, R8, R9, R10, R11, R12, R13, R14, R15),
    REP8(RV_EL, R16, R17, R18, R19, R20, R21, R22, R23),
    REP8(RV_EL, R24, R25, R26, R27, R28, R29, R30, R31),
    /* GPR Aliases */
    REP1(RV_EL, ZERO) = 0,
    REP7(RV_EL, RA, SP, GP, TP, T0, T1, T2),
    REP8(RV_EL, FP, S1, A0, A1, A2, A3, A4, A5),
    REP8(RV_EL, A6, A7, S2, S3, S4, S5, S6, S7),
    REP8(RV_EL, S8, S9, S10, S11, T3, T4, T5, T6),
#if defined(CFG_SISA_RVF) || defined(CFG_SISA_RVD) || defined(CFG_SISA_RVQ)
    /* FPR */
    REP1(RV_EL, FP_S),
    REP1(RV_EL, F0) = REP1(RV_EL, FP_S),
    REP7(RV_EL, F1, F2, F3, F4, F5, F6, F7),
    REP8(RV_EL, F8, F9, F10, F11, F12, F13, F14, F15),
    REP8(RV_EL, F16, F17, F18, F19, F20, F21, F22, F23),
    REP8(RV_EL, F24, F25, F26, F27, F28, F29, F30, F31),
    /* FPR Aliases */
    REP1(RV_EL, FT0) = REP1(RV_EL, FP_S),
    REP7(RV_EL, FT1, FT2, FT3, FT4, FT5, FT6, FT7),
    REP2(RV_EL, FS0, FS1),
    REP8(RV_EL, FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7),
    REP8(RV_EL, FS2, FS3, FS4, FS5, FS6, FS7, FS8, FS9),
    REP2(RV_EL, FS10, FS11),
    REP4(RV_EL, FT8, FT9, FT10, FT11),
#endif
);

RegDef_def(RV,
    [RV_R0]     = { .id = RV_R0             , .name = "r0"  , .alias = "zero"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R1]     = { .id = RV_R1             , .name = "r1"  , .alias = "ra"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R2]     = { .id = RV_R2             , .name = "r2"  , .alias = "sp"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R3]     = { .id = RV_R3             , .name = "r3"  , .alias = "gp"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R4]     = { .id = RV_R4             , .name = "r4"  , .alias = "tp"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R5]     = { .id = RV_R5             , .name = "r5"  , .alias = "t0"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R6]     = { .id = RV_R6             , .name = "r6"  , .alias = "t1"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R7]     = { .id = RV_R7             , .name = "r7"  , .alias = "t2"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R8]     = { .id = RV_R8             , .name = "r8"  , .alias = "fp"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R9]     = { .id = RV_R9             , .name = "r9"  , .alias = "s1"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R10]    = { .id = RV_R10            , .name = "r10" , .alias = "a0"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R11]    = { .id = RV_R11            , .name = "r11" , .alias = "a1"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R12]    = { .id = RV_R12            , .name = "r12" , .alias = "a2"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R13]    = { .id = RV_R13            , .name = "r13" , .alias = "a3"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R14]    = { .id = RV_R14            , .name = "r14" , .alias = "a4"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R15]    = { .id = RV_R15            , .name = "r15" , .alias = "a5"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R16]    = { .id = RV_R16            , .name = "r16" , .alias = "a6"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R17]    = { .id = RV_R17            , .name = "r17" , .alias = "a7"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R18]    = { .id = RV_R18            , .name = "r18" , .alias = "s2"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R19]    = { .id = RV_R19            , .name = "r19" , .alias = "s3"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R20]    = { .id = RV_R20            , .name = "r20" , .alias = "s4"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R21]    = { .id = RV_R21            , .name = "r21" , .alias = "s5"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R22]    = { .id = RV_R22            , .name = "r22" , .alias = "s6"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R23]    = { .id = RV_R23            , .name = "r23" , .alias = "s7"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R24]    = { .id = RV_R24            , .name = "r24" , .alias = "s8"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R25]    = { .id = RV_R25            , .name = "r25" , .alias = "s9"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R26]    = { .id = RV_R26            , .name = "r26" , .alias = "s10"    , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R27]    = { .id = RV_R27            , .name = "r27" , .alias = "s11"    , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R28]    = { .id = RV_R28            , .name = "r28" , .alias = "t3"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R29]    = { .id = RV_R29            , .name = "r29" , .alias = "t4"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R30]    = { .id = RV_R30            , .name = "r30" , .alias = "t5"     , .map = {CFG_SISA_BIT-1, 0} },
    [RV_R31]    = { .id = RV_R31            , .name = "r31" , .alias = "t6"     , .map = {CFG_SISA_BIT-1, 0} },
#if defined(CFG_SISA_RVF) || defined(CFG_SISA_RVD) || defined(CFG_SISA_RVQ)
    [RV_F0]     = { .id = RV_F0-RV_FP_S     , .name = "f0"   , .alias = "ft0"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F1]     = { .id = RV_F1-RV_FP_S     , .name = "f1"   , .alias = "ft1"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F2]     = { .id = RV_F2-RV_FP_S     , .name = "f2"   , .alias = "ft2"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F3]     = { .id = RV_F3-RV_FP_S     , .name = "f3"   , .alias = "ft3"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F4]     = { .id = RV_F4-RV_FP_S     , .name = "f4"   , .alias = "ft4"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F5]     = { .id = RV_F5-RV_FP_S     , .name = "f5"   , .alias = "ft5"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F6]     = { .id = RV_F6-RV_FP_S     , .name = "f6"   , .alias = "ft6"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F7]     = { .id = RV_F7-RV_FP_S     , .name = "f7"   , .alias = "ft7"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F8]     = { .id = RV_F8-RV_FP_S     , .name = "f8"   , .alias = "fs0"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F9]     = { .id = RV_F9-RV_FP_S     , .name = "f9"   , .alias = "fs1"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F10]    = { .id = RV_F10-RV_FP_S    , .name = "f10"  , .alias = "fa0"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F11]    = { .id = RV_F11-RV_FP_S    , .name = "f11"  , .alias = "fa1"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F12]    = { .id = RV_F12-RV_FP_S    , .name = "f12"  , .alias = "fa2"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F13]    = { .id = RV_F13-RV_FP_S    , .name = "f13"  , .alias = "fa3"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F14]    = { .id = RV_F14-RV_FP_S    , .name = "f14"  , .alias = "fa4"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F15]    = { .id = RV_F15-RV_FP_S    , .name = "f15"  , .alias = "fa5"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F16]    = { .id = RV_F16-RV_FP_S    , .name = "f16"  , .alias = "fa6"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F17]    = { .id = RV_F17-RV_FP_S    , .name = "f17"  , .alias = "fa7"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F18]    = { .id = RV_F18-RV_FP_S    , .name = "f18"  , .alias = "fs2"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F19]    = { .id = RV_F19-RV_FP_S    , .name = "f19"  , .alias = "fs3"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F20]    = { .id = RV_F20-RV_FP_S    , .name = "f20"  , .alias = "fs4"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F21]    = { .id = RV_F21-RV_FP_S    , .name = "f21"  , .alias = "fs5"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F22]    = { .id = RV_F22-RV_FP_S    , .name = "f22"  , .alias = "fs6"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F23]    = { .id = RV_F23-RV_FP_S    , .name = "f23"  , .alias = "fs7"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F24]    = { .id = RV_F24-RV_FP_S    , .name = "f24"  , .alias = "fs8"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F25]    = { .id = RV_F25-RV_FP_S    , .name = "f25"  , .alias = "fs9"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F26]    = { .id = RV_F26-RV_FP_S    , .name = "f26"  , .alias = "fs10"  , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F27]    = { .id = RV_F27-RV_FP_S    , .name = "f27"  , .alias = "fs11"  , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F28]    = { .id = RV_F28-RV_FP_S    , .name = "f28"  , .alias = "ft8"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F29]    = { .id = RV_F29-RV_FP_S    , .name = "f29"  , .alias = "ft9"   , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F30]    = { .id = RV_F30-RV_FP_S    , .name = "f30"  , .alias = "ft10"  , .map = {CFG_SISA_BIT-1, 0} },
    [RV_F31]    = { .id = RV_F31-RV_FP_S    , .name = "f31"  , .alias = "ft11"  , .map = {CFG_SISA_BIT-1, 0} },
#endif
);

// ==================================================================================== //
//                                    rv: Insn                                      
// ==================================================================================== //

InsnID_def(RV, 
    REP1(RV_EL, NOP),
#ifdef CFG_SISA_RVI
    /* RV32I: R-Type Arithmetic */
    REP5(RV_EL, ADD, SUB, XOR, OR , AND),
    REP5(RV_EL, SLL, SRL, SRA, SLT, SLTU),
    /* RV32I: I-Type Arithmetic */
    REP4(RV_EL, ADDI, XORI, ORI, ANDI),
    REP5(RV_EL, SLLI, SRLI, SRAI, SLTI, SLTIU),
    /* RV32I: U-Type Arithmetic */
    REP2(RV_EL, LUI, AUIPC),
    /* RV32I: Load I-Type & Store S-Type */
    REP5(RV_EL, LB, LH, LW, LBU, LHU),
    REP3(RV_EL, SB, SH, SW),
    /* RV32I: Branch B-Type & Jump J-Type */
    REP6(RV_EL, BEQ, BNE, BLT, BGE, BLTU, BGEU),
    REP2(RV_EL, JALR, JAL),
    /* RV32I: Device & System */
    REP1(RV_EL, FENCE),
    REP2(RV_EL, ECALL, EBREAK),
#if CFG_SISA_BIT == 64
    /* RV64I: Arithmetic */
    REP5(RV_EL, ADDW, SUBW, SLLW, SRLW, SRAW),
    REP4(RV_EL, ADDIW, SLLIW, SRAIW, SRLIW),
    /* RV64I: Load I-Type & Store S-Type */
    REP3(RV_EL, LD, LWU, SD),
#endif
#endif
#ifdef CFG_SISA_RVZifencei
    /* RV32/64Zifencei: CSR */
    REP1(RV_EL, FENCEI),
#endif
#ifdef CFG_SISA_RVZicsr
    /* RV32/64Zicsr: CSR */
    REP6(RV_EL, CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI),
#endif
#ifdef CFG_SISA_RVM
    /* RV32/64M: Multiply & Divide */
    REP4(RV_EL, MUL, MULH, MULHSU, MULHU),
    REP4(RV_EL, DIV, DIVU, REM, REMU),
    REP5(RV_EL, MULW, DIVW, DIVUW, REMW, REMUW),
#endif
#ifdef CFG_SISA_RVA
    /* RV32/64A: Atomic */
    REP4(RV_EL, LRW, SCW, LRD, SCD),
    REP5(RV_EL, AMOADDW, AMOSWAPW, AMOXORW, AMOORW, AMOANDW),
    REP4(RV_EL, AMOMINW, AMOMAXW, AMOMINUW, AMOMAXUW),
    REP5(RV_EL, AMOADDD, AMOSWAPD, AMOXORD, AMOORD, AMOANDD),
    REP4(RV_EL, AMOMIND, AMOMAXD, AMOMINUD, AMOMAXUD),
#endif
#ifdef CFG_SISA_RVF
    /* RV32/64F: Float Arithmetic */
    REP5(RV_EL, FLW, FSW, FMVXW, FMVWX, FCLASSS),
    REP4(RV_EL, FMADDS, FMSUBS, FNMSUBS, FNMADDS),
    REP5(RV_EL, FADDS, FSUBS, FMULS, FDIVS, FSQRTS),
    REP5(RV_EL, FSGNJS, FSGNJNS, FSGNJXS, FMINS, FMAXS),
    REP3(RV_EL, FLES, FLTS, FEQS),
    REP4(RV_EL, FCVTWS, FCVTWUS, FCVTLS, FCVTLUS),
    REP4(RV_EL, FCVTSW, FCVTSWU, FCVTSL, FCVTSLU),
#endif
#ifdef CFG_SISA_RVD
    /* RV32/64D: Double Arithmetic */
    REP5(RV_EL, FLD, FSD, FMVXD, FMVDX, FCLASSD),
    REP4(RV_EL, FMADDD, FMSUBD, FNMSUBD, FNMADDD),
    REP5(RV_EL, FADDD, FSUBD, FMULD, FDIVD, FSQRTD),
    REP5(RV_EL, FSGNJD, FSGNJND, FSGNJXD, FMIND, FMAXD),
    REP5(RV_EL, FLED, FLT, FEQD, FCVTSD, FCVTDS),
    REP4(RV_EL, FCVTWD, FCVTWUD, FCVTLD, FCVTLUD),
    REP4(RV_EL, FCVTDW, FCVTDWU, FCVTDL, FCVTDLU),
#endif
);


#define Ty_rvrd(V)      Ty_r(V, {11,  7})
#define Ty_rvr1(V)      Ty_r(V, {19, 15})
#define Ty_rvr2(V)      Ty_r(V, {24, 20})
#define Ty_rvii(V)      Ty_i(V, {31, 20})
#define Ty_rvis(V)      Ty_i(V, {11,  7}, {31, 25})
#define Ty_rvib(V)      Ty_i(V, {11,  8}, {30, 25}, { 7,  7}, {31, 31})
#define Ty_rviu(V)      Ty_i(V, {31, 12})
#define Ty_rvij(V)      Ty_i(V, {30, 21}, {20, 20}, {19, 12}, {31, 31})
#define Tys_rvR(...)    Tys_new(Ty_rvrd({}), Ty_rvr1({}), Ty_rvr2({}))
#define Tys_rvI(...)    Tys_new(Ty_rvrd({}), Ty_rvr1({}), Ty_rvii({}))
#define Tys_rvS(...)    Tys_new(Ty_rvr1({}), Ty_rvr2({}), Ty_rvis({}))
#define Tys_rvB(...)    Tys_new(Ty_rvr1({}), Ty_rvr2({}), Ty_rvib({}))
#define Tys_rvU(...)    Tys_new(Ty_rvrd({}), Ty_rviu({}))
#define Tys_rvJ(...)    Tys_new(Ty_rvrd({}), Ty_rvij({}))
#define Tys_rvIe()      Tys_new(Ty_rvrd({}), Ty_rvr1({}))
#define Tys_rv_ri()     Tys_new(Ty_or(Ty_or(Ty_rvib({}), r, {}, {22, 1}), r, {}, {21, 3}))

InsnDef_def(RV,
    [RV_NOP]    = { .id = RV_NOP    , .name = "nop"     , .bc = Val_u32(0x00) },
    /* RV32I: R-Type Arithmetic */
    [RV_ADD]    = { .id = RV_ADD    , .name = "add"     , .bc = Val_u32(0b0110011 + (0b000 << 12))                  , .tv = Tys_rvR()  },
    [RV_SUB]    = { .id = RV_SUB    , .name = "sub"     , .bc = Val_u32(0b0110011 + (0b000 << 12) + (0x20 << 25))   , .tv = Tys_rvR()  },
    [RV_XOR]    = { .id = RV_XOR    , .name = "xor"     , .bc = Val_u32(0b0110011 + (0b100 << 12))                  , .tv = Tys_rvR()  },
    [RV_OR]     = { .id = RV_OR     , .name = "or"      , .bc = Val_u32(0b0110011 + (0b110 << 12))                  , .tv = Tys_rvR()  },
    [RV_AND]    = { .id = RV_AND    , .name = "and"     , .bc = Val_u32(0b0110011 + (0b111 << 12))                  , .tv = Tys_rvR()  },
    [RV_SLL]    = { .id = RV_SLL    , .name = "sll"     , .bc = Val_u32(0b0110011 + (0b001 << 12))                  , .tv = Tys_rvR()  },
    [RV_SRL]    = { .id = RV_SRL    , .name = "srl"     , .bc = Val_u32(0b0110011 + (0b101 << 12))                  , .tv = Tys_rvR()  },
    [RV_SRA]    = { .id = RV_SRA    , .name = "sra"     , .bc = Val_u32(0b0110011 + (0b101 << 12) + (0x20 << 25))   , .tv = Tys_rvR()  },
    [RV_SLT]    = { .id = RV_SLT    , .name = "slt"     , .bc = Val_u32(0b0110011 + (0b010 << 12))                  , .tv = Tys_rvR()  },
    [RV_SLTU]   = { .id = RV_SLTU   , .name = "sltu"    , .bc = Val_u32(0b0110011 + (0b011 << 12))                  , .tv = Tys_rvR()  },
    /* RV32I: I-Type Arithmetic */ 
    [RV_ADDI]   = { .id = RV_ADDI   , .name = "addi"    , .bc = Val_u32(0b0010011 + (0b000 << 12))                  , .tv = Tys_rvI()  },
    [RV_XORI]   = { .id = RV_XORI   , .name = "xori"    , .bc = Val_u32(0b0010011 + (0b100 << 12))                  , .tv = Tys_rvI()  },
    [RV_ORI]    = { .id = RV_ORI    , .name = "ori"     , .bc = Val_u32(0b0010011 + (0b110 << 12))                  , .tv = Tys_rvI()  },
    [RV_ANDI]   = { .id = RV_ANDI   , .name = "andi"    , .bc = Val_u32(0b0010011 + (0b111 << 12))                  , .tv = Tys_rvI()  },
    [RV_SLLI]   = { .id = RV_SLLI   , .name = "slli"    , .bc = Val_u32(0b0010011 + (0b001 << 12))                  , .tv = Tys_rvI()  },
    [RV_SRLI]   = { .id = RV_SRLI   , .name = "srli"    , .bc = Val_u32(0b0010011 + (0b101 << 12))                  , .tv = Tys_rvI()  },
    [RV_SRAI]   = { .id = RV_SRAI   , .name = "srai"    , .bc = Val_u32(0b0010011 + (0b101 << 12) + (0x20 << 25))   , .tv = Tys_rvI()  },
    [RV_SLTI]   = { .id = RV_SLTI   , .name = "slti"    , .bc = Val_u32(0b0010011 + (0b010 << 12))                  , .tv = Tys_rvI()  },
    [RV_SLTIU]  = { .id = RV_SLTIU  , .name = "sltiu"   , .bc = Val_u32(0b0010011 + (0b011 << 12))                  , .tv = Tys_rvI()  },
    /* RV32I: U-Type Arithmetic */
    [RV_LUI]    = { .id = RV_LUI    , .name = "lui"     , .bc = Val_u32(0b0110111)                                  , .tv = Tys_rvU()  },
    [RV_AUIPC]  = { .id = RV_AUIPC  , .name = "auipc"   , .bc = Val_u32(0b0010111)                                  , .tv = Tys_rvU()  },
    /* RV32I: Load I-Type */
    [RV_LB]     = { .id = RV_LB     , .name = "lb"      , .bc = Val_u32(0b0000011 + (0b000 << 12))                  , .tv = Tys_rvI()  },
    [RV_LH]     = { .id = RV_LH     , .name = "lh"      , .bc = Val_u32(0b0000011 + (0b001 << 12))                  , .tv = Tys_rvI()  },
    [RV_LW]     = { .id = RV_LW     , .name = "lw"      , .bc = Val_u32(0b0000011 + (0b010 << 12))                  , .tv = Tys_rvI()  },
    [RV_LBU]    = { .id = RV_LBU    , .name = "lbu"     , .bc = Val_u32(0b0000011 + (0b100 << 12))                  , .tv = Tys_rvI()  },
    [RV_LHU]    = { .id = RV_LHU    , .name = "lhu"     , .bc = Val_u32(0b0000011 + (0b101 << 12))                  , .tv = Tys_rvI()  },
    /* RV32I: Store S-Type */
    [RV_SB]     = { .id = RV_SB     , .name = "sb"      , .bc = Val_u32(0b0100011 + (0b000 << 12))                  , .tv = Tys_rvS()  },
    [RV_SH]     = { .id = RV_SH     , .name = "sh"      , .bc = Val_u32(0b0100011 + (0b001 << 12))                  , .tv = Tys_rvS()  },
    [RV_SW]     = { .id = RV_SW     , .name = "sw"      , .bc = Val_u32(0b0100011 + (0b010 << 12))                  , .tv = Tys_rvS()  },
    /* RV32I: Branch B-Type */
    [RV_BEQ]    = { .id = RV_BEQ    , .name = "beq"     , .bc = Val_u32(0b1100011 + (0b000 << 12))                  , .tv = Tys_rvB()  },
    [RV_BNE]    = { .id = RV_BNE    , .name = "bne"     , .bc = Val_u32(0b1100011 + (0b001 << 12))                  , .tv = Tys_rvB()  },
    [RV_BLT]    = { .id = RV_BLT    , .name = "blt"     , .bc = Val_u32(0b1100011 + (0b100 << 12))                  , .tv = Tys_rvB()  },
    [RV_BGE]    = { .id = RV_BGE    , .name = "bge"     , .bc = Val_u32(0b1100011 + (0b101 << 12))                  , .tv = Tys_rvB()  },
    [RV_BLTU]   = { .id = RV_BLTU   , .name = "bltu"    , .bc = Val_u32(0b1100011 + (0b110 << 12))                  , .tv = Tys_rvB()  },
    [RV_BGEU]   = { .id = RV_BGEU   , .name = "bgeu"    , .bc = Val_u32(0b1100011 + (0b111 << 12))                  , .tv = Tys_rvB()  },
    /* RV32I: Jump J-Type */
    [RV_JALR]   = { .id = RV_JALR   , .name = "jalr"    , .bc = Val_u32(0b1100111 + (0b000 << 12))                  , .tv = Tys_rvI()  },
    [RV_JAL]    = { .id = RV_JAL    , .name = "jal"     , .bc = Val_u32(0b1101111)                                  , .tv = Tys_rvJ()  },
    /* RV32I: Device & System */
    [RV_ECALL]  = { .id = RV_ECALL  , .name = "ecall"   , .bc = Val_u32(0b1110011 + (0b000 << 12) + (0x00 << 20))   , .tv = Tys_rvIe() },
    [RV_EBREAK] = { .id = RV_EBREAK , .name = "ebreak"  , .bc = Val_u32(0b1110011 + (0b000 << 12) + (0x01 << 20))   , .tv = Tys_rvIe() },
);


// Insn_def(RV,

// ,


// );


// ==================================================================================== //
//                                    rv: CPUState                                      
// ==================================================================================== //

CPUState_def(RV,
    CONCAT(u, CFG_SISA_BIT) pc;
    CONCAT(u, CFG_SISA_BIT) gpr[MUXDEF(CFG_SISA_RVE, 16, 32)];
#if defined(CFG_SISA_RVF) || defined(CFG_SISA_RVD) || defined(CFG_SISA_RVQ)
    CONCAT(u, CFG_SISA_BIT) fpr[MUXDEF(CFG_SISA_RVE, 16, 32)];
#endif
,

);


#endif // _ISA_RV_DEF_H_