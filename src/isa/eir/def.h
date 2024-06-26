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
);


RegDef_def(EIR,

);


InsnID_def(EIR, 
    REP1(EIR_EL, ADD_I32),
);

InsnDef_def(EIR,
    [EIR_ADD_I32]   = { .id = EIR_ADD_I32   , .name = "add_i32" , .bc = Val_u32(0x01) , .tr = Tys_new() },
);


Insn_def(EIR

,

);



#endif