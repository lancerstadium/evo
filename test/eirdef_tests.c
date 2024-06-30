#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/eir/def.h>

UnitTest_fn_def(test_insn_display){
    char buf[24];
    for (size_t i = 0; i < InsnMax(EIR); i++) {
        InsnDef_displayone(EIR, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}


UnitTest_fn_def(test_reg_display){
    char buf[24];
    for (size_t i = 0; i < RegMax(EIR); i++) {
        RegDef_displayone(EIR, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}

UnitTest_fn_def(test_insn_encode){
    Insn(EIR)* insn = Insn_new(EIR, EIR_ADD_I32);
    Val* args[3];
    char buf[48];
    args[0] = Val_new_u8(1);
    args[1] = Val_new_u8(2);
    args[2] = Val_new_u8(3);
    Insn_display(EIR, insn, buf);
    UnitTest_msg("%s", buf);
    Insn_encode(EIR, insn, args);
    Insn_display(EIR, insn, buf);
    UnitTest_msg("%s", buf);
    return NULL;
}



UnitTest_fn_def(all_tests) {
    UnitTest_add(test_insn_display);
    UnitTest_add(test_reg_display);
    UnitTest_add(test_insn_encode);
    return NULL;
}

UnitTest_run(all_tests);