#include <isa/rv/def.h>


#if defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/eir/def.h>


#define Tys_gen1op2(ID, OP1, OP2)       Tys_new(Ty_N(ID), Ty_t(OP1), Ty_t(OP2))
#define Tys_gen1op3(ID, OP1, OP2, OP3)  Tys_new(Ty_N(ID), Ty_t(OP1), Ty_t(OP2), Ty_t(OP3))


TransDef_fn_def(RV, EIR,
    {   SID_new(RV_ADD)         , .tt = Tys_gen1op3(EIR_ADD_I32, 0, 1, 2)   },
    {   SID_new(RV_SUB)         , .tt = Tys_gen1op3(EIR_SUB_I32, 0, 1, 2)   },
    {   SID_new(RV_XOR)         , .tt = Tys_gen1op3(EIR_XOR_I32, 0, 1, 2)   },
    {   SID_new(RV_LUI)         , .tt = Tys_gen1op2(EIR_MOV_I32, 0, 1)      },
    {   SID_new(RV_AUIPC)       , .tt = Tys_new(Ty_N(EIR_MOV_I32), Ty_t(0), Ty_gnp(0), Ty_s(0)) },
);




#endif  // CFG_XISA_EIR
