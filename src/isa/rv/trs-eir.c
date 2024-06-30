#include <isa/rv/def.h>


#if defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/eir/def.h>


TransDef_def(RV, EIR,
    {   SID_new(RV_ADD)       , .tys = Tys_new(Ty_N(EIR_ADD_I32)) },
    {   SID_new(RV_SUB)       , .tys = Tys_new(Ty_N(EIR_ADD_I32)) },





);


#endif  // CFG_XISA_EIR
