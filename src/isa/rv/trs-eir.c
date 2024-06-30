#include <isa/rv/trs-eir.h>

#if defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/eir/def.h>


Translator_fn_def(RV, EIR);

#define Tys_gen1op2(ID, OP1, OP2)               Tys_new(Ty_N(ID), Ty_t(OP1), Ty_t(OP2))
#define Tys_gen1op3(ID, OP1, OP2, OP3)          Tys_new(Ty_N(ID), Ty_t(OP1), Ty_t(OP2), Ty_t(OP3))
#define Tys_gencond(OP1, OP2, OP3, COND)        Tys_new(Ty_N(EIR_CMP_I32), Ty_c(COND), Ty_t(OP1), Ty_t(OP2), Ty_t(OP3))
#define Tys_genbrc(COND)                        Tys_new(Ty_nl(0), Ty_gr(1), Ty_gr(2), Ty_gnp(0), Ty_N(EIR_CMP_I32), Ty_c(COND), Ty_s(0), Ty_s(1), Ty_sl(0), Ty_N(EIR_GOTO), Ty_i(1)) 

TransDef_fn_def(RV, EIR,
    /* RV32I: R-Type Arithmetic */
    {   SID_new(RV_ADD)         , .tt = Tys_gen1op3(EIR_ADD_I32 , 0, 1, 2)      },
    {   SID_new(RV_SUB)         , .tt = Tys_gen1op3(EIR_SUB_I32 , 0, 1, 2)      },
    {   SID_new(RV_XOR)         , .tt = Tys_gen1op3(EIR_XOR_I32 , 0, 1, 2)      },
    {   SID_new(RV_OR)          , .tt = Tys_gen1op3(EIR_OR_I32  , 0, 1, 2)      },
    {   SID_new(RV_AND)         , .tt = Tys_gen1op3(EIR_AND_I32 , 0, 1, 2)      },
    {   SID_new(RV_SLL)         , .tt = Tys_gen1op3(EIR_SHL_I32 , 0, 1, 2)      },
    {   SID_new(RV_SRL)         , .tt = Tys_gen1op3(EIR_SHR_I32 , 0, 1, 2)      },
    {   SID_new(RV_SRA)         , .tt = Tys_gen1op3(EIR_SAR_I32 , 0, 1, 2)      },
    {   SID_new(RV_SLT)         , .tt = Tys_gencond(0, 1, 2, TY_COND_LT)        },
    {   SID_new(RV_SLTU)        , .tt = Tys_gencond(0, 1, 2, TY_COND_LTU)       },
    /* RV32I: I-Type Arithmetic */
    {   SID_new(RV_ADDI)        , .tt = Tys_gen1op3(EIR_ADD_I32 , 0, 1, 2)      },
    {   SID_new(RV_XORI)        , .tt = Tys_gen1op3(EIR_XOR_I32 , 0, 1, 2)      },
    {   SID_new(RV_ORI)         , .tt = Tys_gen1op3(EIR_OR_I32  , 0, 1, 2)      },
    {   SID_new(RV_ANDI)        , .tt = Tys_gen1op3(EIR_AND_I32 , 0, 1, 2)      },
    {   SID_new(RV_SLLI)        , .tt = Tys_gen1op3(EIR_SHL_I32 , 0, 1, 2)      },
    {   SID_new(RV_SRLI)        , .tt = Tys_gen1op3(EIR_SHR_I32 , 0, 1, 2)      },
    {   SID_new(RV_SRAI)        , .tt = Tys_gen1op3(EIR_SAR_I32 , 0, 1, 2)      },
    /* RV32I: U-Type Arithmetic */
    {   SID_new(RV_LUI)         , .tt = Tys_gen1op2(EIR_MOV_I32 , 0, 1)         },
    {   SID_new(RV_AUIPC)       , .tt = Tys_new(Ty_gnp(0), Ty_N(EIR_MOV_I32), Ty_t(0), Ty_s(0)) },
    /* RV32I: Load I-Type & Store S-Type */
    {   SID_new(RV_LB)          , .tt = Tys_gen1op3(EIR_LDB_I32 , 0, 1, 2)      },
    {   SID_new(RV_LH)          , .tt = Tys_gen1op3(EIR_LDH_I32 , 0, 1, 2)      },
    {   SID_new(RV_LW)          , .tt = Tys_gen1op3(EIR_LDW_I32 , 0, 1, 2)      },
    {   SID_new(RV_LBU)         , .tt = Tys_gen1op3(EIR_LDB_U32 , 0, 1, 2)      },
    {   SID_new(RV_LHU)         , .tt = Tys_gen1op3(EIR_LDH_U32 , 0, 1, 2)      },
    {   SID_new(RV_SB)          , .tt = Tys_gen1op3(EIR_STB_I32 , 0, 1, 2)      },
    {   SID_new(RV_SH)          , .tt = Tys_gen1op3(EIR_STH_I32 , 0, 1, 2)      },
    {   SID_new(RV_SW)          , .tt = Tys_gen1op3(EIR_STW_I32 , 0, 1, 2)      },
    /* RV32I: Branch B-Type & Jump J-Type */
    {   SID_new(RV_BEQ)         , .tt = Tys_genbrc(TY_COND_EQ)                  },
    {   SID_new(RV_BNE)         , .tt = Tys_genbrc(TY_COND_NE)                  },
    {   SID_new(RV_BLT)         , .tt = Tys_genbrc(TY_COND_LT)                  },
    {   SID_new(RV_BGE)         , .tt = Tys_genbrc(TY_COND_GE)                  },
    {   SID_new(RV_BLTU)        , .tt = Tys_genbrc(TY_COND_LTU)                 },
    {   SID_new(RV_BGEU)        , .tt = Tys_genbrc(TY_COND_GEU)                 },

);


#endif  // CFG_XISA_EIR
