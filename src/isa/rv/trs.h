#ifndef _ISA_RV_TRS_H_
#define _ISA_RV_TRS_H_

#include <evo/cfg.h>

// ==================================================================================== //
//                                    rv: Trans                                     
// ==================================================================================== //

#if defined(CFG_IISA_EIR) || defined(CFG_TISA_EIR)
#include <isa/rv/trs-eir.h>
#endif  // CFG_XISA_EIR

#if defined(CFG_IISA_X86) || defined(CFG_TISA_X86)
#include <isa/rv/trs-x86.h>
#endif  // CFG_XISA_EIR



#endif // _ISA_RV_TRS_H_