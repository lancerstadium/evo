
#include <isa/arm/def.h>

RegDef_fn_def(ARM);

// ==================================================================================== //
//                                    arm: Insn                                      
// ==================================================================================== //

InsnDef_fn_def(ARM);

Insn_fn_def(ARM);


Insn(ARM) * Insn_OP_def(ARM, encode)(Insn(ARM) * insn, Val * args[]) {
    Log_ast(insn != NULL, "Insn: insn is null");
    Log_ast(args != NULL, "Insn: args are null");
    InsnDef(ARM)* df = INSN(ARM, insn->id);
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
Insn(ARM) * Insn_OP_def(ARM, decode)(Val* bc) {
    Insn(ARM)* insn = Insn_OP(ARM, match)(bc);
    if (insn != NULL) {
        InsnDef(ARM)* df = INSN(ARM, insn->id);
        for (size_t i = 0; i < insn->len; i++) {
            BitMap* bm = (df->tr.t[i]).map;
            size_t bml = (df->tr.t[i]).len;
            insn->oprs[i] = Val_ext_map(bc, bm, bml);
            Val_copy(insn->bc, bc);
        }
    }
    return insn;
}
