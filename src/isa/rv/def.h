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

#include <evo/reg.h>
#include <evo/ins.h>
#include <evo/cpu.h>


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
    [RV_R1]     = { .id = RV_R1     , .name = "r1"  , .alias = "ra"   }
);

InsnID_def(RV, 
    REP1(RV_EL, NOP),
    REP1(RV_EL, ADD),
);

InsnDef_def(RV,
    [RV_NOP]    = { .id = RV_NOP    , .name = "nop" , .code = "0x00" },
    [RV_ADD]    = { .id = RV_ADD    , .name = "add" , .code = "0x01" , .tv = TyVec(TYPE_REG_ID, TYPE_REG_ID, TYPE_REG_ID) },
);

CPUState_def(RV,

);


#endif // _ISA_RV_DEF_H_