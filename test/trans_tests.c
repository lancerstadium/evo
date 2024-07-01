#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/trs.h>



UnitTest_fn_def(test_tran_display){
    char buf[24];
    for (size_t i = 0; TransTblEnd(RV, EIR, i) ; i++) {
        TransDef_displayone(RV, EIR, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}


UnitTest_fn_def(test_tran_run){
    char buf[48];
    Translator(RV, EIR) * t = Translator_init(RV, EIR);
    Block(RV) *bb = Block_init(RV);
    Val* val[3];
    val[0] = Val_new_u8(1);
    val[1] = Val_new_u8(3);
    val[2] = Val_new_u8(2);
    Block_push(RV, bb, RV_ADD, val);
    Block_display(RV, bb, buf);
    UnitTest_msg("%s", buf);
    Block(EIR) *tb = Translator_run(RV, EIR, t, bb);
    Block_display(EIR, tb, buf);
    UnitTest_msg("%s", buf);
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_tran_display);
    UnitTest_add(test_tran_run);
    return NULL;
}

UnitTest_run(all_tests);