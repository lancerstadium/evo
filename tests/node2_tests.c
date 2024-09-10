#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_gather) {
    float heat[40] = {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.9, 0.8,
        0.7, 0.6, 0.5, 0.3,
        0.3, 0.2, 0.5, 0.7,

        0.2, 0.2, 0.3, 0.4,
        0.2, 0.8, 0.3, 0.5,
        0.2, 0.3, 1.0, 0.5,
        0.2, 0.4, 0.9, 0.5,
        0.2, 0.4, 0.6, 0.5,
    };
    tensor_t * ts = tensor_new_float32("heat", (int[]){2, 1, 5, 4}, 4, heat, 40);
    tensor_t * ts_i = tensor_new_int64("gather_idx", (int[]){2, 3}, 2, (int64_t[]){0, 2, 1, 3, 0, 1}, 6);
    tensor_t * ts_g = tensor_gather(ts, ts_i, 3);
    tensor_dump2(ts);
    tensor_dump2(ts_i);
    tensor_dump2(ts_g);
}

UnitTest_fn_def(test_all) {
    device_reg("cpu");
    UnitTest_add(test_gather);
    return NULL;
}

UnitTest_run(test_all);