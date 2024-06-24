/**
 * @file isa/eir/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief evo ir defination
 * @date 2024-06-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <evo/evo.h>


#define EIR_EL(I)    EIR_##I

InsnID_def(EIR, 
    REP1(EIR_EL, NOP),
    REP1(EIR_EL, ADD_I32),
);

InsnDef_def(EIR,
    [EIR_NOP]       = { .id = EIR_NOP       , .mnem = "nop"     , .bc = Val_u32(0x00) },
    [EIR_ADD_I32]   = { .id = EIR_ADD_I32   , .mnem = "add_i32" , .bc = Val_u32(0x01) , .tv = Tys_new() },
);
