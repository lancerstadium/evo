#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/x86/def.h>

UnitTest_fn_def(test_insn_display){
    char buf[24];
    for (size_t i = 0; i < InsnMax(X86); i++) {
        InsnDef_displayone(X86, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}


UnitTest_fn_def(test_reg_display){
    char buf[24];
    for (size_t i = 0; i < RegMax(X86); i++) {
        RegDef_displayone(X86, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}



UnitTest_fn_def(all_tests) {
    // UnitTest_add(test_insn_display);
    // UnitTest_add(test_reg_display);
    return NULL;
}

UnitTest_run(all_tests);