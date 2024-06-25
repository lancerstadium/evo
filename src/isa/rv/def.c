
#include <isa/rv/def.h>
#include <evo/task.h>

// ==================================================================================== //
//                                    rv: Reg                                      
// ==================================================================================== //


RegDef_fn_def(RV);


// ==================================================================================== //
//                                    rv: Insn                                      
// ==================================================================================== //

InsnDef_fn_def(RV);

Insn_fn_def(RV);

// ==================================================================================== //
//                                    rv: CPUState                                      
// ==================================================================================== //




// ==================================================================================== //
//                                    rv: Task                                     
// ==================================================================================== //


Insn(RV) * TaskCtx_OP_ISA_def(Decode, run, RV) (TaskCtx(Decode) *ctx, Val bc) {
    Insn(RV) * insn = Insn_new(RV, 0);
    return insn;
}