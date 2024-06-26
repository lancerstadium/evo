#include <sob/sob.h>
#include <evo/evo.h>

UnitTest_fn_def(test_val_bit){
    Val* val = Val_new_u16(0x4321);
    UnitTest_msg("%s", Val_as_hex(val));
    UnitTest_ast(Val_get_u16(val, 0) == 0x4321, "Val new u16 fail");
    Val* val2 = Val_get_bit(val, 15, 8);
    UnitTest_msg("%s", Val_as_hex(val2));
    UnitTest_ast(Val_get_u8(val2, 0) ==   0x43, "Val get bit fail");
    Val_set_bit(val2, 7, 0, Val_new_u8(0xad));
    UnitTest_msg("%s", Val_as_hex(val2));
    UnitTest_ast(Val_get_u8(val2, 0) ==   0xad, "Val set bit fail");
    return NULL;
}

UnitTest_fn_def(test_val_bitmap){
    Val* val = Val_new_u16(0x4321);
    UnitTest_msg("%s", Val_as_hex(val));
    u64 uu = Val_get_map(val, (BitMap[]){{3, 0}, {11, 8}}, 2);
    UnitTest_msg("0x%lx", uu);
    UnitTest_ast(uu == 0x31, "Val get bitmap fail");
    Val_set_map(val, (BitMap[]){{3, 0}, {11, 8}}, 2, 0xab);
    UnitTest_msg("%s", Val_as_hex(val));
    return NULL;
}

UnitTest_fn_def(all_tests) {
    UnitTest_add(test_val_bit);
    UnitTest_add(test_val_bitmap);
    return NULL;
}

UnitTest_run(all_tests);