
#include <isa/x86/def.h>

// ==================================================================================== //
//                                    x86: Reg                                      
// ==================================================================================== //


RegDef_fn_def(X86);


// ==================================================================================== //
//                                    x86: Insn                                      
// ==================================================================================== //


InsnDef_fn_def(X86);

Insn_fn_def(X86);


Insn(X86) * Insn_OP_def(X86, encode)(Insn(X86) * insn, Val * args[]) {
    Log_ast(insn != NULL, "Insn: insn is null");
    Log_ast(args != NULL, "Insn: args are null");
    InsnDef(X86)* df = INSN(X86, insn->id);
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
Insn(X86) * Insn_OP_def(X86, decode)(Val* bc) {
    Insn(X86)* insn = Insn_OP(X86, match)(bc);
    if (insn != NULL) {
        InsnDef(X86)* df = INSN(X86, insn->id);
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
//                                    x86: Block                                      
// ==================================================================================== //

Block_fn_def(X86);

// ==================================================================================== //
//                                    x86: CPUState                                      
// ==================================================================================== //



