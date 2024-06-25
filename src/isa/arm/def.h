/**
 * @file isa/arm/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief armv8 isa defination
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#ifndef _ISA_ARM_DEF_H_
#define _ISA_ARM_DEF_H_

#include <evo/evo.h>

#define ARM_EL(I)   ARM_##I

// ==================================================================================== //
//                                    arm: Reg                                      
// ==================================================================================== //

RegID_def(ARM,
#if CFG_SISA_BIT == 32
    /* ARM32 GPR */
    REP1(ARM_EL, R0) = 0,
    REP7(ARM_EL, R1, R2, R3, R4, R5, R6, R7),
    REP8(ARM_EL, R8, R9, R10, R11, R12, R13, R14, R15),
    /* ARM32 GPR Alias */

#elif CFG_SISA_BIT == 64
    /* AARCH64 GPR 64-bits */
    REP1(ARM_EL, X0) = 0,
    REP7(ARM_EL, X1, X2, X3, X4, X5, X6, X7),
    REP8(ARM_EL, X8, X9, X10, X11, X12, X13, X14, X15),
    REP8(ARM_EL, X16, X17, X18, X19, X20, X21, X22, X23),
    REP8(ARM_EL, X24, X25, X26, X27, X28, X29, X30, X31),
    /* AARCH64 GPR 32-bits */
    REP1(ARM_EL, W0) = 0,
    REP7(ARM_EL, W1, W2, W3, W4, W5, W6, W7),
    REP8(ARM_EL, W8, W9, W10, W11, W12, W13, W14, W15),
    REP8(ARM_EL, W16, W17, W18, W19, W20, W21, W22, W23),
    REP8(ARM_EL, W24, W25, W26, W27, W28, W29, W30, W31),
#endif
);

RegDef_def(ARM,

);

// ==================================================================================== //
//                                    arm: Insn                                      
// ==================================================================================== //

InsnID_def(ARM,
    REP1(ARM_EL, NOP),
);


InsnDef_def(ARM,
    [ARM_NOP]       = { .id = ARM_NOP    , .name = "nop"     , .bc = Val_u32(0x00) },
);

Insn_def(ARM,

,

);

// ==================================================================================== //
//                                    arm: CPUState                                      
// ==================================================================================== //

CPUState_def(ARM,

,

);


#endif