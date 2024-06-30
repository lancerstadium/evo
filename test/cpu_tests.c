#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>


UnitTest_fn_def(test_cpu_reg){
    CPUState(RV) * cpu = CPUState_init(RV, 56);
    char res_buf[32];
    CPUState_set_reg(RV, cpu, RV_R3, Val_new_u64(0x123456789abcdef));
    Val* r = CPUState_get_reg(RV, cpu, RV_R3);
    UnitTest_msg(" R3= %s", ValHex(r));
#if CFG_SISA_BIT == 64
    UnitTest_ast(Val_as_u64(r, 0) == 0x123456789abcdef, "R3 should be 0x123456789abcdef");
#elif CFG_SISA_BIT == 32
    UnitTest_ast(Val_as_u64(r, 0) == 0x89abcdef, "R3 should be 0x89abcdef");
#endif
    for(size_t i = 0; i < RegMax(RV); i++) {
        CPUState_displayreg(RV, cpu, res_buf, i);
        UnitTest_msg("%s", res_buf);
    }
    return NULL;
}

UnitTest_fn_def(test_cpu_mem){
    CPUState(RV) * cpu = CPUState_init(RV, 56);
    Val* addr1 = Val_new_u32(0 + CFG_MEM_BASE);
    Val* addr2 = Val_new_u32(2 + CFG_MEM_BASE);
    CPUState_set_mem(RV, cpu, addr1, Val_new_u64(0x123456), 8);
    Val* val = CPUState_get_mem(RV, cpu, addr2, 4);
    UnitTest_ast(Val_as_u32(val, 0) == 0x12, "Mem[2] should be 0x12");
    UnitTest_msg("%s", ValHex(val));
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
    // UnitTest_add(test_cpu_reg);
    // UnitTest_add(test_cpu_display);
    // UnitTest_add(test_cpu_mem);
    return NULL;
}

UnitTest_run(all_tests);