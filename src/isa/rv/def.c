
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

CPUState_fn_def(RV);

Val* CPUState_OP_def(RV, fetch)(CPUState(RV) * cpu) {
    Val* res = CPUState_get_mem(RV, cpu, cpu->snpc, 4);
    Val_inc(cpu->snpc, res->len);
    return res;
}

Insn(RV) * CPUState_OP_def(RV, decode)(CPUState(RV) * cpu, Val * val) {
    cpu->dnpc = cpu->snpc;
    Insn(RV) * insn = Insn_decode(RV, val);
    return insn;
}


#define RV_EXEC_U(...) \
    u8  rd   = Val_as_u8(insn->oprs[0], 0);  \
    u64 immu = Val_as_u64(insn->oprs[1], 0); \
    Val* res = Val_new_u64(__VA_ARGS__); \
    CPUState_set_reg(RV, cpu, rd, res); \

void CPUState_OP_def(RV, execute)(CPUState(RV) * cpu, Insn(RV) * insn) {
    switch(insn->id) {
        case RV_AUIPC: RV_EXEC_U(Val_as_u64(cpu->pc, 0) + immu); break;
        default: break;
    }
}

// ==================================================================================== //
//                                    rv: Task                                     
// ==================================================================================== //


Insn(RV) * TaskCtx_OP_ISA_def(Exec, run, RV) (UNUSED TaskCtx(Exec) *ctx, UNUSED Val* bc) {
    // Match 
    Insn(RV) * insn = Insn_match(RV, bc);
    if(insn != NULL) {
        UNUSED InsnDef(RV) * def = INSN(RV, insn->id);
        
    }
    return insn;
}