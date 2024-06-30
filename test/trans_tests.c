#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>



UnitTest_fn_def(test_tran_display){
    char buf[24];
    for (size_t i = 0; TransTblEnd(RV, EIR, i) ; i++) {
        TransDef_displayone(RV, EIR, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}




UnitTest_fn_def(all_tests) {
    UnitTest_add(test_tran_display);
    return NULL;
}

UnitTest_run(all_tests);