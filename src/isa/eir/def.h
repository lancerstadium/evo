/**
 * @file isa/eir/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief evo ir defination
 * @date 2024-06-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#ifndef _ISA_EIR_DEF_H_
#define _ISA_EIR_DEF_H_

#include <evo/evo.h>


#define EIR_EL(I)    EIR_##I


RegID_def(EIR,
    REP8(EIR_EL, R0, R1, R2, R3, R4, R5, R6, R7),
    REP8(EIR_EL, R8, R9, R10, R11, R12, R13, R14, R15),
    REP8(EIR_EL, R16, R17, R18, R19, R20, R21, R22, R23),
    REP8(EIR_EL, R24, R25, R26, R27, R28, R29, R30, R31),

);


RegDef_def(EIR,
    [EIR_R0]    = { .id = EIR_R0    , .name = "r0"    , .alias = "r0"    , .map = {63, 0} },
    [EIR_R1]    = { .id = EIR_R1    , .name = "r1"    , .alias = "r1"    , .map = {63, 0} },
    [EIR_R2]    = { .id = EIR_R2    , .name = "r2"    , .alias = "r2"    , .map = {63, 0} },
    [EIR_R3]    = { .id = EIR_R3    , .name = "r3"    , .alias = "r3"    , .map = {63, 0} },
    [EIR_R4]    = { .id = EIR_R4    , .name = "r4"    , .alias = "r4"    , .map = {63, 0} },
    [EIR_R5]    = { .id = EIR_R5    , .name = "r5"    , .alias = "r5"    , .map = {63, 0} },
    [EIR_R6]    = { .id = EIR_R6    , .name = "r6"    , .alias = "r6"    , .map = {63, 0} },
    [EIR_R7]    = { .id = EIR_R7    , .name = "r7"    , .alias = "r7"    , .map = {63, 0} },
    [EIR_R8]    = { .id = EIR_R8    , .name = "r8"    , .alias = "r8"    , .map = {63, 0} },
    [EIR_R9]    = { .id = EIR_R9    , .name = "r9"    , .alias = "r9"    , .map = {63, 0} },
    [EIR_R10]   = { .id = EIR_R10   , .name = "r10"   , .alias = "r10"   , .map = {63, 0} },
    [EIR_R11]   = { .id = EIR_R11   , .name = "r11"   , .alias = "r11"   , .map = {63, 0} },
    [EIR_R12]   = { .id = EIR_R12   , .name = "r12"   , .alias = "r12"   , .map = {63, 0} },
    [EIR_R13]   = { .id = EIR_R13   , .name = "r13"   , .alias = "r13"   , .map = {63, 0} },
    [EIR_R14]   = { .id = EIR_R14   , .name = "r14"   , .alias = "r14"   , .map = {63, 0} },
    [EIR_R15]   = { .id = EIR_R15   , .name = "r15"   , .alias = "r15"   , .map = {63, 0} },
    [EIR_R16]   = { .id = EIR_R16   , .name = "r16"   , .alias = "r16"   , .map = {63, 0} },
    [EIR_R17]   = { .id = EIR_R17   , .name = "r17"   , .alias = "r17"   , .map = {63, 0} },
    [EIR_R18]   = { .id = EIR_R18   , .name = "r18"   , .alias = "r18"   , .map = {63, 0} },
    [EIR_R19]   = { .id = EIR_R19   , .name = "r19"   , .alias = "r19"   , .map = {63, 0} },
    [EIR_R20]   = { .id = EIR_R20   , .name = "r20"   , .alias = "r20"   , .map = {63, 0} },
    [EIR_R21]   = { .id = EIR_R21   , .name = "r21"   , .alias = "r21"   , .map = {63, 0} },
    [EIR_R22]   = { .id = EIR_R22   , .name = "r22"   , .alias = "r22"   , .map = {63, 0} },
    [EIR_R23]   = { .id = EIR_R23   , .name = "r23"   , .alias = "r23"   , .map = {63, 0} },
    [EIR_R24]   = { .id = EIR_R24   , .name = "r24"   , .alias = "r24"   , .map = {63, 0} },
    [EIR_R25]   = { .id = EIR_R25   , .name = "r25"   , .alias = "r25"   , .map = {63, 0} },
    [EIR_R26]   = { .id = EIR_R26   , .name = "r26"   , .alias = "r26"   , .map = {63, 0} },
    [EIR_R27]   = { .id = EIR_R27   , .name = "r27"   , .alias = "r27"   , .map = {63, 0} },
    [EIR_R28]   = { .id = EIR_R28   , .name = "r28"   , .alias = "r28"   , .map = {63, 0} },
    [EIR_R29]   = { .id = EIR_R29   , .name = "r29"   , .alias = "r29"   , .map = {63, 0} },
    [EIR_R30]   = { .id = EIR_R30   , .name = "r30"   , .alias = "r30"   , .map = {63, 0} },
    [EIR_R31]   = { .id = EIR_R31   , .name = "r31"   , .alias = "r31"   , .map = {63, 0} },
);


InsnID_def(EIR, 
    /* ==== EIR: Arithmetic  ===== */
    REP8(EIR_EL, ADD_I32, ADD_I64, SUB_I32, SUB_I64, NEG_I32, NEG_I64, MUL_I32, MUL_I64),
    REP8(EIR_EL, DIV_I32, DIV_I64, DIV_U32, DIV_U64, REM_I32, REM_I64, REM_U32, REM_U64),
    /* ==== EIR: Logical ========= */
    REP8(EIR_EL, AND_I32, AND_I64, OR_I32 , OR_I64 , XOR_I32, XOR_I64, NOT_I32, NOT_I64),
    REP8(EIR_EL, ANDC_I32, ANDC_I64, EQV_I32, EQV_I64, NAND_I32, NAND_I64, NOR_I32, NOR_I64),
    REP6(EIR_EL, ORC_I32, ORC_I64, CLZ_I32, CLZ_I64, CTZ_I32, CTZ_I64),
    /* ==== EIR: Shift =========== */
    REP6(EIR_EL, SHL_I32, SHL_I64, SHR_I32, SHR_I64, SAR_I32, SAR_I64),
    REP4(EIR_EL, ROL_I32, ROL_I64, ROR_I32, ROR_I64),
    /* ==== EIR: Bitwise ========= */
    REP4(EIR_EL, EXTB_I32, EXTB_I64, EXTB_U32, EXTB_U64),
    REP4(EIR_EL, EXTH_I32, EXTH_I64, EXTH_U32, EXTH_U64),
    REP4(EIR_EL, EXTW_I32, EXTW_I64, EXTW_U32, EXTW_U64),
    /* ==== EIR: Move ============ */
    REP2(EIR_EL, MOV_I32, MOV_I64),
    /* ==== EIR: Load & Store ==== */
    REP4(EIR_EL, LDB_I32, LDB_I64, LDB_U32, LDB_U64),
    REP4(EIR_EL, LDH_I32, LDH_I64, LDH_U32, LDH_U64),
    REP4(EIR_EL, LDW_I32, LDW_I64, LDW_U32, LDW_U64),
    REP4(EIR_EL, STB_I32, STB_I64, STH_I32, STH_I64),
    REP2(EIR_EL, STW_I32, STW_I64),
    /* ==== EIR: Compare ========= */
    REP2(EIR_EL, CMP_I32, CMP_I64),
    /* ==== EIR: Function ======== */
    REP2(EIR_EL, CALL, RET),
    /* ==== EIR: System ========= */
    REP3(EIR_EL, TRAP, GOTO, EXIT),
);



/**
 * 
 * ``` 
 *
 * 
 * ```
 */

#define Ty_eirop(V)         Ty_I(V, { 7,  0})
#define Ty_eirrd(V)         Ty_r(V, {15,  8})
#define Ty_eirr1(V)         Ty_r(V, {23, 16})
#define Ty_eirr2(V)         Ty_r(V, {31, 24})
#define Ty_eiriw(V)         Ty_i(V, {63, 32})
#define Ty_eiriw2(V)        Ty_i(V, {55, 24})
#define Ty_eirid(V)         Ty_i(V, {95, 32})
#define Ty_eirid2(V)        Ty_i(V, {87, 24})

#define Tys_eirop()         Tys_new(Ty_eirop(0))
#define Tys_eirIw()         Tys_new(Ty_eirrd(0), Ty_eirr1(0), Ty_eiriw(0))
#define Tys_eirId()         Tys_new(Ty_eirrd(0), Ty_eirr1(0), Ty_eirid(0))
#define Tys_eirRIw()        Tys_new(Ty_eirrd(0), Ty_eirr1(0), Ty_or(Ty_eiriw(0), r, 0, {31, 24}))
#define Tys_eirRIw2()       Tys_new(Ty_eirrd(0), Ty_or(Ty_eiriw(0), r, 0, {23, 16}))
#define Tys_eirRIw4()       Tys_new(Ty_eirrd(0), Ty_eirr1(0), Ty_or(Ty_eiriw(0), r, 0, {31, 24}), Ty_c(0))
#define Tys_eirRId()        Tys_new(Ty_eirrd(0), Ty_eirr1(0), Ty_or(Ty_eirid(0), r, 0, {31, 24}))
#define Tys_eirRId2()       Tys_new(Ty_eirrd(0), Ty_or(Ty_eirid(0), r, 0, {23, 16}))
#define Tys_eirRId4()       Tys_new(Ty_eirrd(0), Ty_eirr1(0), Ty_or(Ty_eirid(0), r, 0, {31, 24}), Ty_c(0))


InsnDef_def(EIR,
    /* ==== EIR: Arithmetic  ===== */
    [EIR_ADD_I32]   = { .id = EIR_ADD_I32   , .mnem = "add_i32" , .bc = Val_u8(0x01)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_ADD_I64]   = { .id = EIR_ADD_I64   , .mnem = "add_i64" , .bc = Val_u8(0x01)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_SUB_I32]   = { .id = EIR_SUB_I32   , .mnem = "sub_i32" , .bc = Val_u8(0x02)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_SUB_I64]   = { .id = EIR_SUB_I64   , .mnem = "sub_i64" , .bc = Val_u8(0x02)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_NEG_I32]   = { .id = EIR_NEG_I32   , .mnem = "neg_i32" , .bc = Val_u8(0x03)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw2() },
    [EIR_NEG_I64]   = { .id = EIR_NEG_I64   , .mnem = "neg_i64" , .bc = Val_u8(0x03)    , .tc = Tys_eirop()     , .tr = Tys_eirRId2() },
    [EIR_MUL_I32]   = { .id = EIR_MUL_I32   , .mnem = "mul_i32" , .bc = Val_u8(0x04)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_MUL_I64]   = { .id = EIR_MUL_I64   , .mnem = "mul_i64" , .bc = Val_u8(0x04)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_DIV_I32]   = { .id = EIR_DIV_I32   , .mnem = "div_i32" , .bc = Val_u8(0x05)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_DIV_I64]   = { .id = EIR_DIV_I64   , .mnem = "div_i64" , .bc = Val_u8(0x05)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_REM_I32]   = { .id = EIR_REM_I32   , .mnem = "rem_i32" , .bc = Val_u8(0x06)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_REM_I64]   = { .id = EIR_REM_I64   , .mnem = "rem_i64" , .bc = Val_u8(0x06)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    /* ==== EIR: Logical ========= */
    [EIR_AND_I32]   = { .id = EIR_AND_I32   , .mnem = "and_i32" , .bc = Val_u8(0x07)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_AND_I64]   = { .id = EIR_AND_I64   , .mnem = "and_i64" , .bc = Val_u8(0x07)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_OR_I32]    = { .id = EIR_OR_I32    , .mnem = "or_i32"  , .bc = Val_u8(0x08)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_OR_I64]    = { .id = EIR_OR_I64    , .mnem = "or_i64"  , .bc = Val_u8(0x08)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_XOR_I32]   = { .id = EIR_XOR_I32   , .mnem = "xor_i32" , .bc = Val_u8(0x09)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_XOR_I64]   = { .id = EIR_XOR_I64   , .mnem = "xor_i64" , .bc = Val_u8(0x09)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_NOT_I32]   = { .id = EIR_NOT_I32   , .mnem = "not_i32" , .bc = Val_u8(0x0A)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw2() },
    [EIR_NOT_I64]   = { .id = EIR_NOT_I64   , .mnem = "not_i64" , .bc = Val_u8(0x0A)    , .tc = Tys_eirop()     , .tr = Tys_eirRId2() },
    /* ==== EIR: Shift =========== */
    [EIR_SHL_I32]   = { .id = EIR_SHL_I32   , .mnem = "shl_i32" , .bc = Val_u8(0x0B)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_SHL_I64]   = { .id = EIR_SHL_I64   , .mnem = "shl_i64" , .bc = Val_u8(0x0B)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_SHR_I32]   = { .id = EIR_SHR_I32   , .mnem = "shr_i32" , .bc = Val_u8(0x0C)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_SHR_I64]   = { .id = EIR_SHR_I64   , .mnem = "shr_i64" , .bc = Val_u8(0x0C)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    [EIR_SAR_I32]   = { .id = EIR_SAR_I32   , .mnem = "sar_i32" , .bc = Val_u8(0x0D)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw()  },
    [EIR_SAR_I64]   = { .id = EIR_SAR_I64   , .mnem = "sar_i64" , .bc = Val_u8(0x0D)    , .tc = Tys_eirop()     , .tr = Tys_eirRId()  },
    /* ==== EIR: Move ============ */
    [EIR_MOV_I32]   = { .id = EIR_MOV_I32   , .mnem = "mov_i32" , .bc = Val_u8(0x0E)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw2() },
    [EIR_MOV_I64]   = { .id = EIR_MOV_I64   , .mnem = "mov_i64" , .bc = Val_u8(0x0E)    , .tc = Tys_eirop()     , .tr = Tys_eirRId2() },
    /* ==== EIR: Load & Store ==== */
    [EIR_LDB_I32]   = { .id = EIR_LDB_I32   , .mnem = "ldb_i32" , .bc = Val_u8(0x0F)    , .tc = Tys_eirop()     , .tr = Tys_eirIw()   },
    [EIR_LDB_I64]   = { .id = EIR_LDB_I64   , .mnem = "ldb_i64" , .bc = Val_u8(0x0F)    , .tc = Tys_eirop()     , .tr = Tys_eirId()   },
    [EIR_LDH_I32]   = { .id = EIR_LDH_I32   , .mnem = "ldh_i32" , .bc = Val_u8(0x0F)    , .tc = Tys_eirop()     , .tr = Tys_eirIw()   },
    [EIR_LDH_I64]   = { .id = EIR_LDH_I64   , .mnem = "ldh_i64" , .bc = Val_u8(0x0F)    , .tc = Tys_eirop()     , .tr = Tys_eirId()   },
    [EIR_LDW_I32]   = { .id = EIR_LDW_I32   , .mnem = "ldw_i32" , .bc = Val_u8(0x0F)    , .tc = Tys_eirop()     , .tr = Tys_eirIw()   },
    [EIR_LDW_I64]   = { .id = EIR_LDW_I64   , .mnem = "ldw_i64" , .bc = Val_u8(0x0F)    , .tc = Tys_eirop()     , .tr = Tys_eirId()   },
    [EIR_STB_I32]   = { .id = EIR_STB_I32   , .mnem = "stb_i32" , .bc = Val_u8(0x10)    , .tc = Tys_eirop()     , .tr = Tys_eirIw()   },
    [EIR_STB_I64]   = { .id = EIR_STB_I64   , .mnem = "stb_i64" , .bc = Val_u8(0x10)    , .tc = Tys_eirop()     , .tr = Tys_eirId()   },
    [EIR_STH_I32]   = { .id = EIR_STH_I32   , .mnem = "sth_i32" , .bc = Val_u8(0x10)    , .tc = Tys_eirop()     , .tr = Tys_eirIw()   },
    [EIR_STH_I64]   = { .id = EIR_STH_I64   , .mnem = "sth_i64" , .bc = Val_u8(0x10)    , .tc = Tys_eirop()     , .tr = Tys_eirId()   },
    [EIR_STW_I32]   = { .id = EIR_STW_I32   , .mnem = "stw_i32" , .bc = Val_u8(0x10)    , .tc = Tys_eirop()     , .tr = Tys_eirIw()   },
    [EIR_STW_I64]   = { .id = EIR_STW_I64   , .mnem = "stw_i64" , .bc = Val_u8(0x10)    , .tc = Tys_eirop()     , .tr = Tys_eirId()   },
    /* ==== EIR: Compare ========= */
    [EIR_CMP_I32]   = { .id = EIR_CMP_I32   , .mnem = "cmp_i32" , .bc = Val_u8(0x11)    , .tc = Tys_eirop()     , .tr = Tys_eirRIw4() },
    [EIR_CMP_I64]   = { .id = EIR_CMP_I64   , .mnem = "cmp_i64" , .bc = Val_u8(0x11)    , .tc = Tys_eirop()     , .tr = Tys_eirRId4()  },

);


Insn_def(EIR

,

);



#endif