/**
 * @file isa/eir/def.h
 * @author LancerStadium (lancerstadium@163.com)
 * @brief evo ir defination
 * @date 2024-06-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <evo/ins.h>


#define EIR_EL(I)    EIR_##I

InsnID_def(EIR, 
    REP1(EIR_EL, NOP),
    REP1(EIR_EL, ADD_I32),
);

InsnDef_def(EIR,
    [EIR_NOP]       = { .id = EIR_NOP       , .name = "nop"     , .bv = ByVec(0x00) },
    [EIR_ADD_I32]   = { .id = EIR_ADD_I32   , .name = "add_i32" , .bv = ByVec(0x01) , .tv = TyVec(TYPE_REG_ID, TYPE_REG_ID, TYPE_REG_ID) },
);
