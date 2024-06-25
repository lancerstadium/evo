/**
 * @file isa/arm/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief armv8 isa defination
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 * @note
 * 
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
    REP8(ARM_EL, X24, X25, X26, X27, X28, X29, X30, XSP),
    REP3(ARM_EL, XZR, SP , PC),
    /* AARCH64 GPR 32-bits */
    REP8(ARM_EL, W0, W1, W2, W3, W4, W5, W6, W7),
    REP8(ARM_EL, W8, W9, W10, W11, W12, W13, W14, W15),
    REP8(ARM_EL, W16, W17, W18, W19, W20, W21, W22, W23),
    REP8(ARM_EL, W24, W25, W26, W27, W28, W29, W30, WSP),
    REP2(ARM_EL, WZR, WSP),
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


#endif // _ISA_ARM_DEF_H_