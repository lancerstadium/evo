#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>


UnitTest_fn_def(test_reg_display){
    char buf[24];
    for (size_t i = 0; i < RegMax(RV); i++) {
        RegDef_displayone(RV, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}


UnitTest_fn_def(test_insn_display){
    char buf[24];
    for (size_t i = 0; i < InsnMax(RV); i++) {
        InsnDef_displayone(RV, buf, i);
        UnitTest_msg("%s", buf);
    }
    return NULL;
}

UnitTest_fn_def(test_insn_decode){
    // Test Website: https://luplab.gitlab.io/rvcodecjs/
    // lbu x10, 16(x5) == lbu x5, x10, 16
    Insn(RV)* insn = Insn_decode(RV, Val_new_u32(0b0110011 + (0b011 << 12) + (0x00 << 25) + (0x02 << 7)));
    UnitTest_msg("%s", InsnName(RV, insn->id));
    UnitTest_ast(insn->id == RV_SLTU, "Match `sltu` fail");
    char buf[48];
    Insn_display(RV, insn, buf);
    UnitTest_msg("%s", buf);                                        
    Insn(RV)* insn2= Insn_decode(RV, Val_new_u32(0x00000297));      /* auipc x5, 0   */
    Insn_display(RV, insn2, buf);
    UnitTest_msg("%s", buf);
    Insn(RV)* insn3= Insn_decode(RV, Val_new_u32(0x00028823));      /* sb x5, x0, 16 */
    Insn_display(RV, insn3, buf);
    UnitTest_msg("%s", buf);
    Insn(RV)* insn4= Insn_decode(RV, Val_new_u32(0x0102c503));      /* lbu x10, x5, 16 */
    Insn_display(RV, insn4, buf);
    UnitTest_msg("%s", buf);
    Insn(RV)* insn5= Insn_match(RV, Val_new_u32(0x00100073));      /* ebreak */
    Insn_display(RV, insn5, buf);
    UnitTest_msg("%s", buf);
    return NULL;
}


UnitTest_fn_def(test_insn_encode){
    Insn(RV)* insn = Insn_new(RV, RV_XOR);
    Val* args[3];
    char buf[48];
    args[0] = Val_new_u8(1);
    args[1] = Val_new_u8(2);
    args[2] = Val_new_u8(3);
    Insn_display(RV, insn, buf);
    UnitTest_msg("%s", buf);
    Insn_encode(RV, insn, args);
    Insn_display(RV, insn, buf);
    UnitTest_msg("%s", buf);
    return NULL;
}



UnitTest_fn_def(all_tests) {
    // UnitTest_add(test_reg_display);
    // UnitTest_add(test_insn_display);
    // UnitTest_add(test_insn_decode);
    // UnitTest_add(test_insn_encode);
    return NULL;
}

UnitTest_run(all_tests);