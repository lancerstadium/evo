
#ifndef _ISA_RV_TRS_EIR_H_
#define _ISA_RV_TRS_EIR_H_


#include <isa/rv/def.h>


#if defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/eir/def.h>


TransDef_def(RV, EIR);
Translator_def(RV, EIR);



#endif  // CFG_XISA_EIR


#endif // _ISA_RV_TRS_EIR_H_