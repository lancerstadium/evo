/**
 * @file isa/arm/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief armv8 isa defination
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 * @note
 * - [Arm Insn](https://groupoid.github.io/languages/ARM.htm)
 * - [A64 Dissamble](https://tonpa.guru/stream/2022/2022-06-11%20A64%20Disassembler.htm)
 * 
 * ## 1 Arm 64 Reg
 * 
 * A64:  R0-R30     -- general purpose registers (63..0: Xn, 31..0: Wn)
 *       R31        -- zero register (63..0)
 *       SP         -- stack pointer (63..0)
 *       PC         -- program counter (63..0)
 * 
 * NEON: V0-V31     -- FPU/SIMD registers (127..0: Qn, 63..0: Dn, 31..0: Sn, 15..0: Hn, 7..0: Bn)
 *       FPCR,FPSR  -- FPU/SIMD status registers (63..0)
 * 
 * SVE:  P0-P15     -- SVE predicate registers (255..0, ..., 15..0)
 *       FFR        -- SVE first fault register (63..0)
 *       Z0-Z31     -- SVE registers (2047..0, ..., 127..0: Qn)
 *                                   (63..0: Dn, 31..0: Sn, 15..0: Hn, 7..0: Bn)
 * 
 * ## 2 Arm 64 Insn Format
 * 
 * - A64 four parts Insntruction:
 *      1. A64  (Main)
 *      2. NEON (FPU/SIMD Extension)
 *      3. SVE  (Vector Extension)
 *      4. SME  (Matrix Extension)
 * 
 * ```txt
 *      +---------------------= 32/64-bit
 *      |
 *      |     0 0 0 0 --------= SME
 *      |     0 0 1 0 --------= SVE
 *      |     1 0 0 x --------= Data Processing Immediate
 *      |     1 0 1 x --------= Branching, System, Exceptions
 *      |     x 1 x 0 --------= Loads and Stores
 *      |     x 1 0 1 --------= Data Processing Register
 *      |     1 1 1 1 --------= NEON Scalar
 *      |     0 1 1 1 --------= NEON Vector
 *      |     | | | |
 *      |     | | | |
 *      7 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0
 *      +-------------+ +-------------+ +-------------+ +-------------+
 *      |     MSB     | |             | |             | |     LSB     |
 *      +-------------+ +-------------+ +-------------+ +-------------+
 * ```
 * 
 * ### 2.1 A64 Main
 * 
 * ```txt
 *      |                31-21  | 20-16: Rm | F | 14-10: Ra | 9-5: Rn   | 4-0: Rd   | Description
 *      +-----------------------+-----------+---+-----------+-----------+-----------+--------------------------
 *      | s q x t t t t z a r w | x x x x x | x | x x x x x | x x x x x | x x x x x | % 0-way PURE :
 *      | s q x t t t t z a r w | x x o o o | x | x x x x x | x x o o o | Rd        | % 1-way REG  :
 *      | s q x t t t t z a r w | Rm        | c | c c c x x | Rn        | x n z c v | % 2-way REG  RIGHT :
 *      | s q x t t t t z a r w | x x x x x | x | x Z x x M | Rn        | Rd        | % 2-way REG  LEFT  :
 *      | s q x t t t t z a r w | Rm        | x | x x x x x | Rn        | Rd        | % 3-way REG  :
 *      | s q x t t t t z a r w | Rm        | x | Ra        | Rn        | Rd        | % 4-way REG  :
 *      | s q x t t t | imm26                                                       | % 0-way BRA  :
 *      | s q x t t t t z a r w | imm5      | c | c c c x x | Rn        | x n z c v | % 1-way REG  IMM   :
 *      | s q x t t t t z a r w | imm16                                 | Rd        | % 1-way REG  BIG   IMM :
 *      | s q x t t t t z | imm19                                       | Rd        | % 1-way REG  LARGE IMM :
 *      | s q x t t t t z a r w | imm9                | x x | Rn        | Rd        | % 2-way REG  IMM   :
 *      | s q x t t t t z a r | imm12                       | Rn        | Rd        | % 2-way REG  BIG   IMM :
 *      | s q x t t t t z a r | immr        | imms          | Rn        | Rd        | % 2-way REG  PAIR  IMM :
 *      | s q x t t t t z a r w | Rm        | imm6          | Rn        | Rd        | % 3-way REG  RIGHT IMM :
 *      | s q x t t t t z a r | imm7            | Ra        | Rn        | Rd        | % 3-way REG  LEFT  IMM :
 * ```
 * 
 * ### 2.2 NEON
 * 
 * ```txt
 *      |                31-21  | 20-16: Rm | F | 14-10: Ra | 9-5: Rn   | 4-0: Rd   | Description
 *      +-----------------------+-----------+---+-----------+-----------+-----------+-------------------
 *      | s q x t t t t z a r w | Rm        | x | x x x x x | Rn        | Rd        | % 3-way REG  :
 *      | s q x t t t t z a r w | a b c d e | x | x x x x x | f g h i j | Rd        | % 1-way REG  :
 *      | s q x t t t t z a r w | imm5      | x | x x x x x | Rn        | Rd        | % 2-way REG  IMM   :
 *      | s a x t t t t z a r w | Rm        | 0 | Ra        | Rn        | Rd        | % 4-way REG  :
 *      | s q x t t t t z a r w | Rm        | 0 | imm4  | 0 | Rn        | Rd        | % 3-way REG  IMM   :
 * ```
 * ### 2.3 SVE
 * 
 * ```txt
 *      |                31-21  | 20-16: Rm | F | 14-10: Ra | 9-5: Rn   | 4-0: Rd   | Description
 *      +-----------------------+-----------+---+-----------+-----------+-----------+-------------------
 *      | s q x t t t t z a r w | Zm        | x | x x x x x | Zn        | Zd        | % 3-Way REG  :
 *      | s q x t t t t z a r w | x x x x x | x | x x | imm8            | Zd        | % 1-Way REG  IMM   :
 * ```
 * 
 * ### 2.4 SME
 * 
 * ```txt
 *      |                31-21  | 20-16: Rm | F | 14-10: Ra | 9-5: Rn   | 4-0: Rd   | Description
 *      +-----------------------+-----------+---+-----------+-----------+-----------+-------------------
 *      | s q x t t t t z a r w | x x x x x | x | y y z z z | a b c d e | f g h i j | % 0-Way PURE :
 *      | s q x t t t t z a r w | x x x x x | x | y y z z z | Zn        | a b c d e | % 1-Way REG  LEFT  :
 *      | s q x t t t t z a r w | x x x x x | x | y y z z z | a b c d e | Zd        | % 1-Way REG  RIGHT :
 *      | s q x t t t t z a r w | Zm        | x | y y z z z | Zn        | a b c d e | % 2-Way REG  :
 *      | s q x t t t t z a r w | Rn        | x | y y z z | imm6        | Rd        | % 2-Way REG  IMM   :
 * ```
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
    /* ARM32 GPR: R0-R15 */
    REP1(ARM_EL, R0) = 0,
    REP7(ARM_EL, R1, R2, R3, R4, R5, R6, R7),
    REP8(ARM_EL, R8, R9, R10, R11, R12, R13, R14, R15),
    /* ARM32 GPR Alias */
    REP1(ARM_EL, PC) = REP1(ARM_EL, R15),
#elif CFG_SISA_BIT == 64
    /* AARCH64 GPR 64-bits */
    REP1(ARM_EL, X0) = 0,
    REP7(ARM_EL, X1, X2, X3, X4, X5, X6, X7),
    REP8(ARM_EL, X8, X9, X10, X11, X12, X13, X14, X15),
    REP8(ARM_EL, X16, X17, X18, X19, X20, X21, X22, X23),
    REP8(ARM_EL, X24, X25, X26, X27, X28, X29, X30, X31),
    REP4(ARM_EL, XZR, SP , PC , LR),
    /* AARCH64 GPR 32-bits */
    REP8(ARM_EL, W0, W1, W2, W3, W4, W5, W6, W7),
    REP8(ARM_EL, W8, W9, W10, W11, W12, W13, W14, W15),
    REP8(ARM_EL, W16, W17, W18, W19, W20, W21, W22, W23),
    REP8(ARM_EL, W24, W25, W26, W27, W28, W29, W30, W31),
    REP2(ARM_EL, WZR, WSP),
#endif
);

RegDef_def(ARM,
#if CFG_SISA_BIT == 32
    [ARM_R0]    = { .id = ARM_R0            , .name = "r0"  , .alias = "r0"     , .map = {31, 0} },
    [ARM_R1]    = { .id = ARM_R1            , .name = "r1"  , .alias = "r1"     , .map = {31, 0} },
    [ARM_R2]    = { .id = ARM_R2            , .name = "r2"  , .alias = "r2"     , .map = {31, 0} },
    [ARM_R3]    = { .id = ARM_R3            , .name = "r3"  , .alias = "r3"     , .map = {31, 0} },
    [ARM_R4]    = { .id = ARM_R4            , .name = "r4"  , .alias = "r4"     , .map = {31, 0} },
    [ARM_R5]    = { .id = ARM_R5            , .name = "r5"  , .alias = "r5"     , .map = {31, 0} },
    [ARM_R6]    = { .id = ARM_R6            , .name = "r6"  , .alias = "r6"     , .map = {31, 0} },
    [ARM_R7]    = { .id = ARM_R7            , .name = "r7"  , .alias = "r7"     , .map = {31, 0} },
    [ARM_R8]    = { .id = ARM_R8            , .name = "r8"  , .alias = "r8"     , .map = {31, 0} },
    [ARM_R9]    = { .id = ARM_R9            , .name = "r9"  , .alias = "r9"     , .map = {31, 0} },
    [ARM_R10]   = { .id = ARM_R10           , .name = "r10" , .alias = "r10"    , .map = {31, 0} },
    [ARM_R11]   = { .id = ARM_R11           , .name = "r11" , .alias = "r11"    , .map = {31, 0} },
    [ARM_R12]   = { .id = ARM_R12           , .name = "r12" , .alias = "r12"    , .map = {31, 0} },
    [ARM_R13]   = { .id = ARM_R13           , .name = "r13" , .alias = "r13"    , .map = {31, 0} },
    [ARM_R14]   = { .id = ARM_R14           , .name = "r14" , .alias = "r14"    , .map = {31, 0} },
    [ARM_R15]   = { .id = ARM_R15           , .name = "r15" , .alias = "pc"     , .map = {31, 0} },
#elif CFG_SISA_BIT == 64
    [ARM_X0]    = { .id = ARM_X0            , .name = "x0"  , .alias = "r0"     , .map = {63, 0} },
    [ARM_X1]    = { .id = ARM_X1            , .name = "x1"  , .alias = "r1"     , .map = {63, 0} },
    [ARM_X2]    = { .id = ARM_X2            , .name = "x2"  , .alias = "r2"     , .map = {63, 0} },
    [ARM_X3]    = { .id = ARM_X3            , .name = "x3"  , .alias = "r3"     , .map = {63, 0} },
    [ARM_X4]    = { .id = ARM_X4            , .name = "x4"  , .alias = "r4"     , .map = {63, 0} },
    [ARM_X5]    = { .id = ARM_X5            , .name = "x5"  , .alias = "r5"     , .map = {63, 0} },
    [ARM_X6]    = { .id = ARM_X6            , .name = "x6"  , .alias = "r6"     , .map = {63, 0} },
    [ARM_X7]    = { .id = ARM_X7            , .name = "x7"  , .alias = "r7"     , .map = {63, 0} },
    [ARM_X8]    = { .id = ARM_X8            , .name = "x8"  , .alias = "r8"     , .map = {63, 0} },
    [ARM_X9]    = { .id = ARM_X9            , .name = "x9"  , .alias = "r9"     , .map = {63, 0} },
    [ARM_X10]   = { .id = ARM_X10           , .name = "x10" , .alias = "r10"    , .map = {63, 0} },
    [ARM_X11]   = { .id = ARM_X11           , .name = "x11" , .alias = "r11"    , .map = {63, 0} },
    [ARM_X12]   = { .id = ARM_X12           , .name = "x12" , .alias = "r12"    , .map = {63, 0} },
    [ARM_X13]   = { .id = ARM_X13           , .name = "x13" , .alias = "r13"    , .map = {63, 0} },
    [ARM_X14]   = { .id = ARM_X14           , .name = "x14" , .alias = "r14"    , .map = {63, 0} },
    [ARM_X15]   = { .id = ARM_X15           , .name = "x15" , .alias = "r15"    , .map = {63, 0} },
    [ARM_X16]   = { .id = ARM_X16           , .name = "x16" , .alias = "r16"    , .map = {63, 0} },
    [ARM_X17]   = { .id = ARM_X17           , .name = "x17" , .alias = "r17"    , .map = {63, 0} },
    [ARM_X18]   = { .id = ARM_X18           , .name = "x18" , .alias = "r18"    , .map = {63, 0} },
    [ARM_X19]   = { .id = ARM_X19           , .name = "x19" , .alias = "r19"    , .map = {63, 0} },
    [ARM_X20]   = { .id = ARM_X20           , .name = "x20" , .alias = "r20"    , .map = {63, 0} },
    [ARM_X21]   = { .id = ARM_X21           , .name = "x21" , .alias = "r21"    , .map = {63, 0} },
    [ARM_X22]   = { .id = ARM_X22           , .name = "x22" , .alias = "r22"    , .map = {63, 0} },
    [ARM_X23]   = { .id = ARM_X23           , .name = "x23" , .alias = "r23"    , .map = {63, 0} },
    [ARM_X24]   = { .id = ARM_X24           , .name = "x24" , .alias = "r24"    , .map = {63, 0} },
    [ARM_X25]   = { .id = ARM_X25           , .name = "x25" , .alias = "r25"    , .map = {63, 0} },
    [ARM_X26]   = { .id = ARM_X26           , .name = "x26" , .alias = "r26"    , .map = {63, 0} },
    [ARM_X27]   = { .id = ARM_X27           , .name = "x27" , .alias = "r27"    , .map = {63, 0} },
    [ARM_X28]   = { .id = ARM_X28           , .name = "x28" , .alias = "r28"    , .map = {63, 0} },
    [ARM_X29]   = { .id = ARM_X29           , .name = "x29" , .alias = "r29"    , .map = {63, 0} },
    [ARM_X30]   = { .id = ARM_X30           , .name = "x30" , .alias = "r30"    , .map = {63, 0} },
    [ARM_X31]   = { .id = ARM_X31           , .name = "x31" , .alias = "r31"    , .map = {63, 0} },
    [ARM_XZR]   = { .id = ARM_X31           , .name = "xzr" , .alias = "zero"   , .map = {63, 0} },
    [ARM_SP]    = { .id = ARM_X31           , .name = "sp"  , .alias = "sp"     , .map = {63, 0} },
    [ARM_PC]    = { .id = ARM_X31           , .name = "pc"  , .alias = "pc"     , .map = {63, 0} },
    [ARM_LR]    = { .id = ARM_X31           , .name = "lr"  , .alias = "lr"     , .map = {63, 0} },
    [ARM_W0]    = { .id = ARM_X31           , .name = "w0"  , .alias = "r0"     , .map = {31, 0} },
    [ARM_W1]    = { .id = ARM_X1            , .name = "w1"  , .alias = "r1"     , .map = {31, 0} },
    [ARM_W2]    = { .id = ARM_X2            , .name = "w2"  , .alias = "r2"     , .map = {31, 0} },
    [ARM_W3]    = { .id = ARM_X3            , .name = "w3"  , .alias = "r3"     , .map = {31, 0} },
    [ARM_W4]    = { .id = ARM_X4            , .name = "w4"  , .alias = "r4"     , .map = {31, 0} },
    [ARM_W5]    = { .id = ARM_X5            , .name = "w5"  , .alias = "r5"     , .map = {31, 0} },
    [ARM_W6]    = { .id = ARM_X6            , .name = "w6"  , .alias = "r6"     , .map = {31, 0} },
    [ARM_W7]    = { .id = ARM_X7            , .name = "w7"  , .alias = "r7"     , .map = {31, 0} },
    [ARM_W8]    = { .id = ARM_X8            , .name = "w8"  , .alias = "r8"     , .map = {31, 0} },
    [ARM_W9]    = { .id = ARM_X9            , .name = "w9"  , .alias = "r9"     , .map = {31, 0} },
    [ARM_W10]   = { .id = ARM_X10           , .name = "w10" , .alias = "r10"    , .map = {31, 0} },
    [ARM_W11]   = { .id = ARM_X11           , .name = "w11" , .alias = "r11"    , .map = {31, 0} },
    [ARM_W12]   = { .id = ARM_X12           , .name = "w12" , .alias = "r12"    , .map = {31, 0} },
    [ARM_W13]   = { .id = ARM_X13           , .name = "w13" , .alias = "r13"    , .map = {31, 0} },
    [ARM_W14]   = { .id = ARM_X14           , .name = "w14" , .alias = "r14"    , .map = {31, 0} },
    [ARM_W15]   = { .id = ARM_X15           , .name = "w15" , .alias = "r15"    , .map = {31, 0} },
    [ARM_W16]   = { .id = ARM_X16           , .name = "w16" , .alias = "r16"    , .map = {31, 0} },
    [ARM_W17]   = { .id = ARM_X17           , .name = "w17" , .alias = "r17"    , .map = {31, 0} },
    [ARM_W18]   = { .id = ARM_X18           , .name = "w18" , .alias = "r18"    , .map = {31, 0} },
    [ARM_W19]   = { .id = ARM_X19           , .name = "w19" , .alias = "r19"    , .map = {31, 0} },
    [ARM_W20]   = { .id = ARM_X20           , .name = "w20" , .alias = "r20"    , .map = {31, 0} },
    [ARM_W21]   = { .id = ARM_X21           , .name = "w21" , .alias = "r21"    , .map = {31, 0} },
    [ARM_W22]   = { .id = ARM_X22           , .name = "w22" , .alias = "r22"    , .map = {31, 0} },
    [ARM_W23]   = { .id = ARM_X23           , .name = "w23" , .alias = "r23"    , .map = {31, 0} },
    [ARM_W24]   = { .id = ARM_X24           , .name = "w24" , .alias = "r24"    , .map = {31, 0} },
    [ARM_W25]   = { .id = ARM_X25           , .name = "w25" , .alias = "r25"    , .map = {31, 0} },
    [ARM_W26]   = { .id = ARM_X26           , .name = "w26" , .alias = "r26"    , .map = {31, 0} },
    [ARM_W27]   = { .id = ARM_X27           , .name = "w27" , .alias = "r27"    , .map = {31, 0} },
    [ARM_W28]   = { .id = ARM_X28           , .name = "w28" , .alias = "r28"    , .map = {31, 0} },
    [ARM_W29]   = { .id = ARM_X29           , .name = "w29" , .alias = "r29"    , .map = {31, 0} },
    [ARM_W30]   = { .id = ARM_X30           , .name = "w30" , .alias = "r30"    , .map = {31, 0} },
    [ARM_W31]   = { .id = ARM_X31           , .name = "w31" , .alias = "r31"    , .map = {31, 0} },
    [ARM_WZR]   = { .id = ARM_X31           , .name = "wzr" , .alias = "zero"   , .map = {31, 0} },
    [ARM_WSP]   = { .id = ARM_X31           , .name = "wsp" , .alias = "wsp"    , .map = {31, 0} },
#endif
);

// ==================================================================================== //
//                                    arm: Insn                                      
// ==================================================================================== //



InsnID_def(ARM,
    REP1(ARM_EL, UDF),
);


InsnDef_def(ARM,
    [ARM_UDF]       = { .id = ARM_UDF    , .name = "udf"     , .bc = Val_u32(0x00) },
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


#endif // _ISA_ARM_DEF_H_