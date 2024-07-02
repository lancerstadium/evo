
#include <isa/eir/def.h>

// ==================================================================================== //
//                                    eir: Reg                                      
// ==================================================================================== //


RegDef_fn_def(EIR);


// ==================================================================================== //
//                                    eir: Insn                                      
// ==================================================================================== //


InsnDef_fn_def(EIR);

Insn_fn_def(EIR);


Insn(EIR) * Insn_OP_def(EIR, encode)(Insn(EIR) * insn, Val * args[]) {
    Log_ast(insn != NULL, "Insn: insn is null");
    Log_ast(args != NULL, "Insn: args are null");
    InsnDef(EIR)* df = INSN(EIR, insn->id);
    for (size_t i = 0; i < insn->len; i++) {
        if (args[i] != NULL) {
            BitMap* bm = (df->tr.t[i]).map;
            size_t bml = (df->tr.t[i]).len;
            insn->oprs[i] = Val_tymatch(args[i], &df->tr.t[i]);
            Val_imp_map(&insn->bc, bm, bml, args[i]);
        }
    }
    return insn;
}
Insn(EIR) * Insn_OP_def(EIR, decode)(Val* bc) {
    Insn(EIR)* insn = Insn_OP(EIR, match)(bc);
    if (insn != NULL) {
        InsnDef(EIR)* df = INSN(EIR, insn->id);
        for (size_t i = 0; i < insn->len; i++) {
            BitMap* bm = (df->tr.t[i]).map;
            size_t bml = (df->tr.t[i]).len;
            insn->oprs[i] = Val_ext_map(bc, bm, bml);
            Val_copy(insn->bc, bc);
        }
    }
    return insn;
}

// ==================================================================================== //
//                                    eir: Block                                      
// ==================================================================================== //

Block_fn_def(EIR);

// ==================================================================================== //
//                                    eir: CPUState                                      
// ==================================================================================== //
