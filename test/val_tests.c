#include <sob/sob.h>
#include <evo/evo.h>

UnitTest_fn_def(test_val_bit){
    Val* val = Val_new_u16(0x4321);
    UnitTest_msg("%s", ValHex(val));
    UnitTest_ast(Val_get_u16(val, 0) == 0x4321, "Val new u16 fail");
    Val* val2 = Val_get_bit(val, 15, 8);
    UnitTest_msg("%s", ValHex(val2));
    UnitTest_ast(Val_get_u8(val2, 0) ==   0x43, "Val get bit fail");
    Val_set_bit(val, 15, 8, Val_new_u8(0xad));
    UnitTest_msg("%s", ValHex(val));
    UnitTest_ast(Val_get_u8(val, 1)  ==   0xad, "Val set bit fail");
    Val_set_bit(val, 11, 7, Val_new_u8(0xff));
    UnitTest_msg("%s", ValHex(val));
    return NULL;
}

UnitTest_fn_def(test_val_bitmap){
    Val* val = Val_new_u16(0x4351);
    UnitTest_msg("%s", ValHex(val));
    u64 uu = Val_get_map(val, (BitMap[]){{3, 0}, {11, 8}}, 2);
    UnitTest_msg("0x%lx", uu);
    UnitTest_ast(uu == 0x31, "Val get bitmap fail");
    UnitTest_msg("%s", ValBin(val));
    Val_set_map(val, (BitMap[]){{11, 7}, {3, 1}}, 2, 0b11111111);
    UnitTest_msg("%s", ValBin(val));
    UnitTest_ast(Val_get_u8(val, 0)  == 0b11011111, "Val set bitmap fail");
    UnitTest_msg("%s", ValHex(val));
    UnitTest_ast(Val_get_u16(val, 0) == 0x4fdf, "Val set bitmap fail");
    bool res = Val_eq_map(val, (BitMap[]){{11, 7}, {3, 1}}, 2, Val_new_u16(0b11111111));
    UnitTest_ast(res, "Val eq bitmap fail");
    return NULL;
}

UnitTest_fn_def(all_tests) {
    UnitTest_add(test_val_bit);
    UnitTest_add(test_val_bitmap);
    return NULL;
}

UnitTest_run(all_tests);