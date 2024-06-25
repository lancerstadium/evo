#include <sob/sob.h>
#include <evo/evo.h>

UnitTest_fn_def(test_val_bit){
    Val* val = Val_new_u16(0x0123);
    UnitTest_ast(Val_get_u16(val, 0) == 0x0123, "Val u16");
    Val* val2 = Val_get_bit(val, 11, 4);
    UnitTest_msg("%s", Val_as_hex(val2));
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_val_bit);
    return NULL;
}

UnitTest_run(all_tests);