#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>


UnitTest_fn_def(test_cpu_reg){
    CPUState(RV) * cpu = CPUState_init(RV, 56);
    char res_buf[32];
    CPUState_set_reg(RV, cpu, RV_R3, Val_new_u64(0x123456789abcdef));
    Val* r = CPUState_get_reg(RV, cpu, RV_R3);
    UnitTest_msg(" R3= %s", ValHex(r));
    UnitTest_ast(Val_get_u64(r, 0) == 0x123456789abcdef, "R3 should be 0x123456789abcdef");
    for(size_t i = 0; i < RegMax(RV); i++) {
        CPUState_displayreg(RV, cpu, res_buf, i);
        UnitTest_msg("%s", res_buf);
    }
    return NULL;
}

UnitTest_fn_def(test_cpu_display){
    CPUState(RV) * cpu = CPUState_init(RV, 56);
    char res_buf[48];
    CPUState_display(RV, cpu, res_buf);
    UnitTest_msg("%s", res_buf);
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_cpu_reg);
    UnitTest_add(test_cpu_display);
    return NULL;
}

UnitTest_run(all_tests);