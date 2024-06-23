/**
 * @file isa/eir/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief evo ir defination
 * @date 2024-06-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */



#ifndef _ISA_RV_DEF_H_
#define _ISA_RV_DEF_H_

#include <evo/evo.h>


#define RV_EL(I)    RV_##I


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
);

RegDef_def(RV,
    [RV_R0]     = { .id = RV_R0     , .name = "r0"  , .alias = "zero" },
    [RV_R1]     = { .id = RV_R1     , .name = "r1"  , .alias = "ra"   },
    [RV_R2]     = { .id = RV_R2     , .name = "r2"  , .alias = "sp"   },
    [RV_R3]     = { .id = RV_R3     , .name = "r3"  , .alias = "gp"   },
    [RV_R4]     = { .id = RV_R4     , .name = "r4"  , .alias = "tp"   },
    [RV_R5]     = { .id = RV_R5     , .name = "r5"  , .alias = "t0"   },
    [RV_R6]     = { .id = RV_R6     , .name = "r6"  , .alias = "t1"   },
    [RV_R7]     = { .id = RV_R7     , .name = "r7"  , .alias = "t2"   },
    [RV_R8]     = { .id = RV_R8     , .name = "r8"  , .alias = "fp"   },
    [RV_R9]     = { .id = RV_R9     , .name = "r9"  , .alias = "s1"   },
    [RV_R10]    = { .id = RV_R10    , .name = "r10" , .alias = "a0"   },
    [RV_R11]    = { .id = RV_R11    , .name = "r11" , .alias = "a1"   },
    [RV_R12]    = { .id = RV_R12    , .name = "r12" , .alias = "a2"   },
    [RV_R13]    = { .id = RV_R13    , .name = "r13" , .alias = "a3"   },
    [RV_R14]    = { .id = RV_R14    , .name = "r14" , .alias = "a4"   },
    [RV_R15]    = { .id = RV_R15    , .name = "r15" , .alias = "a5"   },
    [RV_R16]    = { .id = RV_R16    , .name = "r16" , .alias = "a6"   },
    [RV_R17]    = { .id = RV_R17    , .name = "r17" , .alias = "a7"   },
    [RV_R18]    = { .id = RV_R18    , .name = "r18" , .alias = "s2"   },
    [RV_R19]    = { .id = RV_R19    , .name = "r19" , .alias = "s3"   },
    [RV_R20]    = { .id = RV_R20    , .name = "r20" , .alias = "s4"   },
    [RV_R21]    = { .id = RV_R21    , .name = "r21" , .alias = "s5"   },
    [RV_R22]    = { .id = RV_R22    , .name = "r22" , .alias = "s6"   },
    [RV_R23]    = { .id = RV_R23    , .name = "r23" , .alias = "s7"   },
    [RV_R24]    = { .id = RV_R24    , .name = "r24" , .alias = "s8"   },
    [RV_R25]    = { .id = RV_R25    , .name = "r25" , .alias = "s9"   },
    [RV_R26]    = { .id = RV_R26    , .name = "r26" , .alias = "s10"  },
    [RV_R27]    = { .id = RV_R27    , .name = "r27" , .alias = "s11"  },
    [RV_R28]    = { .id = RV_R28    , .name = "r28" , .alias = "t3"   },
    [RV_R29]    = { .id = RV_R29    , .name = "r29" , .alias = "t4"   },
    [RV_R30]    = { .id = RV_R30    , .name = "r30" , .alias = "t5"   },
    [RV_R31]    = { .id = RV_R31    , .name = "r31" , .alias = "t6"   },
);

InsnID_def(RV, 
    REP1(RV_EL, NOP),
    /* RV32I: R-Type Arithmetic */
    REP5(RV_EL, ADD, SUB, XOR, OR , AND),
    REP5(RV_EL, SLL, SRL, SRA, SLT, SLTU),
    /* RV32I: I-Type Arithmetic */
    REP4(RV_EL, ADDI, XORI, ORI, ANDI),
    REP5(RV_EL, SLLI, SRLI, SRAI, SLTI, SLTIU),
    /* RV32I: Load & Store */
    REP7(RV_EL, LB, LH, LW, LD, LBU, LHU, LWU),
    REP4(RV_EL, SB, SH, SW, SD),
    /* RV32I: Branch & Jump */
    REP6(RV_EL, BEQ, BNE, BLT, BGE, BLTU, BGEU),
    REP2(RV_EL, JALR, JAL),
    /* RV32I: Misc & Trap */
    REP2(RV_EL, LUI, AUIPC),
    REP2(RV_EL, FENCE, FENCEI),
    REP2(RV_EL, ECALL, EBREAK),
    /* RV64I: Arithmetic */
    REP5(RV_EL, ADDW, SUBW, SLLW, SRLW, SRAW),
    REP4(RV_EL, ADDIW, SLLIW, SRAIW, SRLIW),
    /* RV32/64M: Multiply & Divide */
    REP4(RV_EL, MUL, MULH, MULHSU, MULHU),
    REP4(RV_EL, DIV, DIVU, REM, REMU),
    REP5(RV_EL, MULW, DIVW, DIVUW, REMW, REMUW),
    /* RV32/64A: Atomic */
    REP4(RV_EL, LRW, SCW, LRD, SCD),
    REP5(RV_EL, AMOADDW, AMOSWAPW, AMOXORW, AMOORW, AMOANDW),
    REP4(RV_EL, AMOMINW, AMOMAXW, AMOMINUW, AMOMAXUW),
    REP5(RV_EL, AMOADDD, AMOSWAPD, AMOXORD, AMOORD, AMOANDD),
    REP4(RV_EL, AMOMIND, AMOMAXD, AMOMINUD, AMOMAXUD),
    /* RV32/64Zicsr: CSR */
    REP6(RV_EL, CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI),
    /* RV32/64F: Float Arithmetic */
    REP5(RV_EL, FLW, FSW, FMVXW, FMVWX, FCLASSS),
    REP4(RV_EL, FMADDS, FMSUBS, FNMSUBS, FNMADDS),
    REP5(RV_EL, FADDS, FSUBS, FMULS, FDIVS, FSQRTS),
    REP5(RV_EL, FSGNJS, FSGNJNS, FSGNJXS, FMINS, FMAXS),
    REP3(RV_EL, FLES, FLTS, FEQS),
    REP4(RV_EL, FCVTWS, FCVTWUS, FCVTLS, FCVTLUS),
    REP4(RV_EL, FCVTSW, FCVTSWU, FCVTSL, FCVTSLU),
    /* RV32/64D: Double Arithmetic */
    REP5(RV_EL, FLD, FSD, FMVXD, FMVDX, FCLASSD),
    REP4(RV_EL, FMADDD, FMSUBD, FNMSUBD, FNMADDD),
    REP5(RV_EL, FADDD, FSUBD, FMULD, FDIVD, FSQRTD),
    REP5(RV_EL, FSGNJD, FSGNJND, FSGNJXD, FMIND, FMAXD),
    REP5(RV_EL, FLED, FLT, FEQD, FCVTSD, FCVTDS),
    REP4(RV_EL, FCVTWD, FCVTWUD, FCVTLD, FCVTLUD),
    REP4(RV_EL, FCVTDW, FCVTDWU, FCVTDL, FCVTDLU),
);


InsnDef_def(RV,
    [RV_NOP]    = { .id = RV_NOP    , .name = "nop" , .bc = ByU32(0x00) },
    [RV_ADD]    = { .id = RV_ADD    , .name = "add" , .bc = ByU32(0b110011 + (0b000 << 12))                 , .tv = TyVec(TY_REG_ID, TY_REG_ID, TY_REG_ID) },
    [RV_SUB]    = { .id = RV_SUB    , .name = "sub" , .bc = ByU32(0b110011 + (0b000 << 12) + (0x20 << 25))  , .tv = TyVec(TY_REG_ID, TY_REG_ID, TY_REG_ID) },
    [RV_XOR]    = { .id = RV_XOR    , .name = "xor" , .bc = ByU32(0b110011 + (0b100 << 12))                 , .tv = TyVec(TY_REG_ID, TY_REG_ID, TY_REG_ID) },
);

CPUState_def(RV,
#if CFG_SISA_BIT == 64
    u64 gpr[MUXDEF(CFG_SISA_RVE, 16, 32)];
    u64 pc;
#else
    u32 gpr[MUXDEF(CFG_SISA_RVE, 16, 32)];
    u32 pc;
#endif
,

);


#endif // _ISA_RV_DEF_H_