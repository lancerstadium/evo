#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>

UnitTest_fn_def(test_insn_display){
    char buf[24];
    for (size_t i = 0; i < InsnMax(RV); i++) {
        InsnDef_displayone(RV, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}


UnitTest_fn_def(test_reg_display){
    char buf[24];
    for (size_t i = 0; i < RegMax(RV); i++) {
        RegDef_displayone(RV, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}



UnitTest_fn_def(all_tests) {
    UnitTest_add(test_insn_display);
    UnitTest_add(test_reg_display);
    return NULL;
}

UnitTest_run(all_tests);