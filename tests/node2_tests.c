#include "sob.h"
#include <evo.h>

UnitTest_fn_def(test_gather) {
    float data[40] = {
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
    tensor_t * ts = tensor_new_float32("input", (int[]){2, 1, 5, 4}, 4, data, 40);
    tensor_t * ts_i = tensor_new_int64("gather_idx", (int[]){2, 3}, 2, (int64_t[]){0, 2, 1, 3, 0, 1}, 6);
    tensor_t * ts_g = tensor_gather(ts, ts_i, 3);
    tensor_dump2(ts);
    tensor_dump2(ts_i);
    tensor_dump2(ts_g);
}

UnitTest_fn_def(test_pad) {
    float data[6] = {
        1.0, 1.2,
        2.3, 3.4,
        4.5, 5.7,
    };
    tensor_t * ts = tensor_new_float32("input", (int[]){2, 3}, 2, data, 6);
    tensor_t * ts_p = tensor_pad(ts, (int64_t[]){0, 3, 0, 3}, 4, "reflect");
    tensor_dump2(ts);
    tensor_dump2(ts_p);
}

UnitTest_fn_def(test_scatternd) {
    // float data1[8] = {
    //     1, 2, 3, 4, 5, 6, 7, 8
    // };
    // float data2[4] = {
    //     9, 10, 11, 12
    // };
    // tensor_t * ts = tensor_new_float32("input", (int[]){8}, 1, data1, 8);
    // tensor_t * ts_u = tensor_new_float32("update", (int[]){4}, 1, data2, 4);
    // tensor_t * ts_i = tensor_new_int64("indexs", (int[]){4, 1}, 2, (int64_t[]){4, 3, 1, 7}, 4);
    // tensor_t * ts_p = tensor_scatternd(ts, ts_i, ts_u, "none");
    float data1[64] = {
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8
    };
    float data2[32] = {
        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4
    };
    tensor_t * ts = tensor_new_float32("input", (int[]){4, 4, 4}, 3, data1, 64);
    tensor_t * ts_u = tensor_new_float32("update", (int[]){2, 4, 4}, 3, data2, 32);
    tensor_t * ts_i = tensor_new_int64("indexs", (int[]){2, 1}, 2, (int64_t[]){0, 3}, 2);
    tensor_t * ts_p = tensor_scatternd(ts, ts_i, ts_u, "none");
    // float data1[64] = {
    //     1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
    //     1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
    //     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
    //     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8
    // };
    // float data2[32] = {
    //     1, 1, 1, 1, 
    //     2, 2, 2, 2, 
    //     3, 3, 3, 3, 
    //     4, 4, 4, 4
    // };
    // tensor_t * ts = tensor_new_float32("input", (int[]){4, 4, 4}, 3, data1, 64);
    // tensor_t * ts_u = tensor_new_float32("update", (int[]){4, 4}, 2, data2, 32);
    // tensor_t * ts_i = tensor_new_int64("indexs", (int[]){4, 2}, 2, (int64_t[]){0, 0, 1, 1, 2, 2, 3, 3}, 8);
    // tensor_t * ts_p = tensor_scatternd(ts, ts_i, ts_u, "none");
    tensor_dump2(ts);
    tensor_dump2(ts_u);
    tensor_dump2(ts_p);
    return NULL;
}

UnitTest_fn_def(test_expand) {
    float data[8] = {
        1, 9, 1, 3,
        3, 6, 8, 1
    };
    tensor_t * ts = tensor_new_float32("input", (int[]){2, 1, 4}, 3, data, 8);
    tensor_t * ts_p = tensor_expand(ts, (int64_t[]){2, 2, 3, 4}, 4);
    tensor_dump2(ts);
    tensor_dump2(ts_p);
    return NULL;
}


UnitTest_fn_def(test_quant) {
    float data[8] = {
        1.2, 9.1, -4.2, 19.3,
        -3.3, 6.9, 1.2, 11.9
    };
    tensor_t * ts = tensor_new_float32("input", (int[]){2, 1, 4}, 3, data, 8);
    tensor_t * sc = tensor_new_float32("scale", (int[]){1}, 1, (float[]){1.8}, 1);
    tensor_t * zp = tensor_new("zeropoint", TENSOR_TYPE_INT8);
    tensor_reshape(zp, 1, (int[]){1});
    tensor_apply(zp, (int8_t[]){0}, 1 * sizeof(int8_t));
    tensor_t * ts_p = tensor_quant(ts, sc, zp, 1, TENSOR_TYPE_INT8);
    tensor_t * ts_b = tensor_dequant(ts_p, sc, zp, 1);

    // tensor_dump2(ts);
    tensor_dump2(ts_p);
    tensor_dump2(ts_b);
    return NULL;
}


UnitTest_fn_def(test_all) {
    device_reg("cpu");
    // UnitTest_add(test_gather);
    // UnitTest_add(test_pad);
    // UnitTest_add(test_scatternd);
    // UnitTest_add(test_expand);
    UnitTest_add(test_quant);
    device_unreg("cpu");
    return NULL;
}

UnitTest_run(test_all);