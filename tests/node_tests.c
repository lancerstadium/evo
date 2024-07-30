#include "sob.h"
#include <evo.h>

#define MD(s)       "node/"s"/model.onnx"
#define TI(s, i)    "node/"s"/test_data_set_0/input_"#i".pb"
#define TO(s, i)    "node/"s"/test_data_set_0/output_"#i".pb"

#define TESTD_1I(s, i, o)                       \
    model_t* mdl = sez->load_model(sez, MD(s)); \
    tensor_t* i = model_get_tensor(mdl, #i);    \
    tensor_t* o = model_get_tensor(mdl, #o);    \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));  \
    tensor_t* t1 = sez->load_tensor(TO(s, 0));  \
    tensor_copy(i, t0);                         \
    graph_prerun(mdl->graph);                   \
    graph_run(mdl->graph);                      \
    tensor_dump2(o);                            \
    tensor_dump2(t1);                           \
    mdl->sez->unload(mdl);

#define TEST_1I(s, i, o)                            \
    model_t* mdl = sez->load_model(sez, MD(s));     \
    tensor_t* i = model_get_tensor(mdl, #i);        \
    tensor_t* o = model_get_tensor(mdl, #o);        \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));      \
    tensor_t* t1 = sez->load_tensor(TO(s, 0));      \
    tensor_copy(i, t0);                             \
    graph_prerun(mdl->graph);                       \
    graph_run(mdl->graph);                          \
    UnitTest_ast(tensor_equal(o, t1), s " failed"); \
    mdl->sez->unload(mdl);

#define TESTD_2I(s, i0, i1, o)                  \
    model_t* mdl = sez->load_model(sez, MD(s)); \
    tensor_t* i0 = model_get_tensor(mdl, #i0);  \
    tensor_t* i1 = model_get_tensor(mdl, #i1);  \
    tensor_t* o = model_get_tensor(mdl, #o);    \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));  \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));  \
    tensor_t* t2 = sez->load_tensor(TO(s, 0));  \
    tensor_copy(i0, t0);                        \
    tensor_copy(i1, t1);                        \
    graph_prerun(mdl->graph);                   \
    graph_run(mdl->graph);                      \
    tensor_dump2(o);                            \
    tensor_dump2(t2);                           \
    mdl->sez->unload(mdl);

#define TEST_2I(s, i0, i1, o)                       \
    model_t* mdl = sez->load_model(sez, MD(s));     \
    tensor_t* i0 = model_get_tensor(mdl, #i0);      \
    tensor_t* i1 = model_get_tensor(mdl, #i1);      \
    tensor_t* o = model_get_tensor(mdl, #o);        \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));      \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));      \
    tensor_t* t2 = sez->load_tensor(TO(s, 0));      \
    tensor_copy(i0, t0);                            \
    tensor_copy(i1, t1);                            \
    graph_prerun(mdl->graph);                       \
    graph_run(mdl->graph);                          \
    UnitTest_ast(tensor_equal(o, t2), s " failed"); \
    mdl->sez->unload(mdl);

#define TESTD_3I(s, i0, i1, i2, o)              \
    model_t* mdl = sez->load_model(sez, MD(s)); \
    tensor_t* i0 = model_get_tensor(mdl, #i0);  \
    tensor_t* i1 = model_get_tensor(mdl, #i1);  \
    tensor_t* i2 = model_get_tensor(mdl, #i2);  \
    tensor_t* o = model_get_tensor(mdl, #o);    \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));  \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));  \
    tensor_t* t2 = sez->load_tensor(TI(s, 2));  \
    tensor_t* t3 = sez->load_tensor(TO(s, 0));  \
    tensor_copy(i0, t0);                        \
    tensor_copy(i1, t1);                        \
    tensor_copy(i2, t2);                        \
    graph_prerun(mdl->graph);                   \
    graph_run(mdl->graph);                      \
    tensor_dump2(o);                            \
    tensor_dump2(t3);                           \
    mdl->sez->unload(mdl);

#define TEST_3I(s, i0, i1, i2, o)                   \
    model_t* mdl = sez->load_model(sez, MD(s));     \
    tensor_t* i0 = model_get_tensor(mdl, #i0);      \
    tensor_t* i1 = model_get_tensor(mdl, #i1);      \
    tensor_t* i2 = model_get_tensor(mdl, #i2);      \
    tensor_t* o = model_get_tensor(mdl, #o);        \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));      \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));      \
    tensor_t* t2 = sez->load_tensor(TI(s, 2));      \
    tensor_t* t3 = sez->load_tensor(TO(s, 0));      \
    tensor_copy(i0, t0);                            \
    tensor_copy(i1, t1);                            \
    tensor_copy(i2, t2);                            \
    graph_prerun(mdl->graph);                       \
    graph_run(mdl->graph);                          \
    UnitTest_ast(tensor_equal(o, t3), s " failed"); \
    mdl->sez->unload(mdl);

#define TEST_5I(s, i0, i1, i2, i3, i4, o)           \
    model_t* mdl = sez->load_model(sez, MD(s));     \
    tensor_t* i0 = model_get_tensor(mdl, #i0);      \
    tensor_t* i1 = model_get_tensor(mdl, #i1);      \
    tensor_t* i2 = model_get_tensor(mdl, #i2);      \
    tensor_t* i3 = model_get_tensor(mdl, #i3);      \
    tensor_t* i4 = model_get_tensor(mdl, #i4);      \
    tensor_t* o = model_get_tensor(mdl, #o);        \
    tensor_t* t0 = sez->load_tensor(TI(s, 0));      \
    tensor_t* t1 = sez->load_tensor(TI(s, 1));      \
    tensor_t* t2 = sez->load_tensor(TI(s, 2));      \
    tensor_t* t3 = sez->load_tensor(TI(s, 3));      \
    tensor_t* t4 = sez->load_tensor(TI(s, 4));      \
    tensor_t* t5 = sez->load_tensor(TO(s, 0));      \
    tensor_copy(i0, t0);                            \
    tensor_copy(i1, t1);                            \
    tensor_copy(i2, t2);                            \
    tensor_copy(i3, t3);                            \
    tensor_copy(i4, t4);                            \
    graph_prerun(mdl->graph);                       \
    graph_run(mdl->graph);                          \
    UnitTest_ast(tensor_equal(o, t5), s " failed"); \
    mdl->sez->unload(mdl);

serializer_t * sez;

// ---------------------- Init  ----------------------

UnitTest_fn_def(test_node_init) {
    device_reg("cpu");
    sez = serializer_new("onnx");
    return NULL;
}

// ---------------------- Abs    ----------------------

UnitTest_fn_def(test_abs) {
    TEST_1I("test_abs", x, y);
    return NULL;
}

// ---------------------- Add    ----------------------

UnitTest_fn_def(test_add) {
    TEST_2I("test_add", x, y, sum);
    return NULL;
}

// ---------------------- AveragePool -----------------

UnitTest_fn_def(test_averagepool_1d) {
    TEST_1I("test_averagepool_1d", x, y);
    return NULL;
}
UnitTest_fn_def(test_averagepool_2d) {
    TEST_1I("test_averagepool_2d", x, y);
    return NULL;
}
UnitTest_fn_def(test_averagepool_2dc) {
    TEST_1I("test_averagepool_2d_ceil", x, y);
    return NULL;
}
UnitTest_fn_def(test_averagepool_2dp) {
    TEST_1I("test_averagepool_2d_pads", x, y);
    return NULL;
}
UnitTest_fn_def(test_averagepool_3d) {
    TEST_1I("test_averagepool_3d", x, y);
    return NULL;
}
UnitTest_fn_def(test_averagepool) {
    UnitTest_add(test_averagepool_1d);
    UnitTest_add(test_averagepool_2d);
    UnitTest_add(test_averagepool_2dc);
    UnitTest_add(test_averagepool_2dp);
    UnitTest_add(test_averagepool_3d);
    return NULL;
}

// ---------------------- Batchnorm -------------------

UnitTest_fn_def(test_batchnorm_epsilon) { 
    TEST_5I("test_batchnorm_epsilon", x, s, bias, mean, var, y);
    return NULL; 
}
UnitTest_fn_def(test_batchnorm) {
    UnitTest_add(test_batchnorm_epsilon);
    return NULL;
}

// ---------------------- Cast   ----------------------

UnitTest_fn_def(test_cast_bf16_to_f32) { 
    TEST_1I("test_cast_BFLOAT16_to_FLOAT", input, output);
    return NULL;
}
UnitTest_fn_def(test_cast_f64_to_f32) { 
    TEST_1I("test_cast_DOUBLE_to_FLOAT", input, output);
    return NULL;
}
UnitTest_fn_def(test_cast_f64_to_f16) { 
    TEST_1I("test_cast_DOUBLE_to_FLOAT16", input, output);
    return NULL;
}
UnitTest_fn_def(test_cast) {
    UnitTest_add(test_cast_bf16_to_f32);
    UnitTest_add(test_cast_f64_to_f32);
    UnitTest_add(test_cast_f64_to_f16);
    return NULL;
}

// ---------------------- Concat ----------------------

UnitTest_fn_def(test_concat_1d_a0) { 
    TEST_2I("test_concat_1d_axis_0", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat_1d_an1) { 
    TEST_2I("test_concat_1d_axis_negative_1", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat_2d_a0) { 
    TEST_2I("test_concat_2d_axis_0", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat_2d_a1) { 
    TEST_2I("test_concat_2d_axis_1", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat_3d_a0) { 
    TEST_2I("test_concat_3d_axis_0", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat_3d_a1) { 
    TEST_2I("test_concat_3d_axis_1", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat_3d_a2) { 
    TEST_2I("test_concat_3d_axis_2", value0, value1, output);
    return NULL; 
}
UnitTest_fn_def(test_concat) {
    UnitTest_add(test_concat_1d_a0);
    UnitTest_add(test_concat_1d_an1);
    UnitTest_add(test_concat_2d_a0);
    UnitTest_add(test_concat_2d_a1);
    UnitTest_add(test_concat_3d_a0);
    UnitTest_add(test_concat_3d_a1);
    UnitTest_add(test_concat_3d_a2);
    return NULL;
}

// ---------------------- Conv  -----------------------

UnitTest_fn_def(test_conv_ap) {
    TEST_2I("test_conv_with_autopad_same", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv_sap) {
    TEST_2I("test_conv_with_strides_and_asymmetric_padding", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv_snp) {
    TEST_2I("test_conv_with_strides_no_padding", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv_sp) {
    TEST_2I("test_conv_with_strides_padding", x, W, y);
    return NULL;
}
UnitTest_fn_def(test_conv) {
    UnitTest_add(test_conv_ap);
    UnitTest_add(test_conv_sap);
    UnitTest_add(test_conv_snp);
    UnitTest_add(test_conv_sp);
    return NULL;
}

// ---------------------- Dropout ----------------------

UnitTest_fn_def(test_dropout_dft) {
    TEST_1I("test_dropout_dft", x, y);
    return NULL;
}
UnitTest_fn_def(test_dropout_dft_m) {
    TEST_1I("test_dropout_dft_mask", x, y);
    return NULL;
}
UnitTest_fn_def(test_dropout_dft_mr) {
    TEST_1I("test_dropout_dft_mask_ratio", x, y);
    return NULL;
}
UnitTest_fn_def(test_dropout_dft_o) {
    TEST_1I("test_dropout_dft_old", x, y);
    return NULL;
}
UnitTest_fn_def(test_dropout_dft_r) {
    TEST_1I("test_dropout_dft_ratio", x, y);
    return NULL;
}
UnitTest_fn_def(test_dropout_ro) {
    TEST_1I("test_dropout_random_old", x, y);
    return NULL;
}
UnitTest_fn_def(test_dropout) {
    UnitTest_add(test_dropout_dft);
    UnitTest_add(test_dropout_dft_m);
    UnitTest_add(test_dropout_dft_mr);
    UnitTest_add(test_dropout_dft_o);
    UnitTest_add(test_dropout_dft_r);
    UnitTest_add(test_dropout_ro);
    return NULL;
}

// ---------------------- Gemm    ----------------------


UnitTest_fn_def(test_gemm_aa) {
    TEST_3I("test_gemm_all_attributes", a, b, c, y);
    return NULL;
}
UnitTest_fn_def(test_gemm_a) {
    TEST_3I("test_gemm_alpha", a, b, c, y);
    return NULL;
}
UnitTest_fn_def(test_gemm_b) {
    TESTD_3I("test_gemm_beta", a, b, c, y);
    return NULL;
}
UnitTest_fn_def(test_gemm_dft_mb) {
    TEST_3I("test_gemm_default_matrix_bias", a, b, c, y);
    return NULL;
}
UnitTest_fn_def(test_gemm_dft_nb) {
    TEST_3I("test_gemm_default_no_bias", a, b, c, y);
    return NULL;
}
UnitTest_fn_def(test_gemm_dft_sb) {
    TEST_3I("test_gemm_default_scalar_bias", a, b, c, y);
    return NULL;
}
UnitTest_fn_def(test_gemm) {
    UnitTest_add(test_gemm_aa);
    UnitTest_add(test_gemm_a);
    // UnitTest_add(test_gemm_b);
    // UnitTest_add(test_gemm_dft_mb);
    // UnitTest_add(test_gemm_dft_nb);
    // UnitTest_add(test_gemm_dft_sb);
    return NULL;
}

// ---------------------- GlobalAveragePool ------------

UnitTest_fn_def(test_globalaveragepool_dft) {
    TEST_1I("test_globalaveragepool_dft", x, y);
    return NULL;
}
UnitTest_fn_def(test_globalaveragepool_p) {
    TEST_1I("test_globalaveragepool_precomputed", x, y);
    return NULL;
}
UnitTest_fn_def(test_globalaveragepool) {
    // UnitTest_add(test_globalaveragepool_dft);    // fault
    UnitTest_add(test_globalaveragepool_p);
    return NULL;
}

// ---------------------- MatMul  ----------------------

UnitTest_fn_def(test_matmul_2d) {
    TEST_2I("test_matmul_2d", a, b, c);
    return NULL;
}
UnitTest_fn_def(test_matmul_3d) {
    TEST_2I("test_matmul_3d", a, b, c);
    return NULL;
}
UnitTest_fn_def(test_matmul_4d) {
    TEST_2I("test_matmul_4d", a, b, c);
    return NULL;
}
UnitTest_fn_def(test_matmul) {
    UnitTest_add(test_matmul_2d);
    UnitTest_add(test_matmul_3d);
    UnitTest_add(test_matmul_4d);
    return NULL;
}

// ---------------------- LeakyRelu --------------------

UnitTest_fn_def(test_leakyrelu_a) {
    TEST_1I("test_leakyrelu_alpha", x, y);
    return NULL;
}
UnitTest_fn_def(test_leakyrelu_dft) {
    TEST_1I("test_leakyrelu_dft", x, y);
    return NULL;
}
UnitTest_fn_def(test_leakyrelu_ex) {
    TEST_1I("test_leakyrelu_example", x, y);
    return NULL;
}
UnitTest_fn_def(test_leakyrelu) {
    UnitTest_add(test_leakyrelu_a);
    // UnitTest_add(test_leakyrelu_dft);
    UnitTest_add(test_leakyrelu_ex);
    return NULL;
}

// ---------------------- MaxPool ----------------------

UnitTest_fn_def(test_maxpool_1d) {
    TEST_1I("test_maxpool_1d", x, y);   
    return NULL;
}
UnitTest_fn_def(test_maxpool_2d) {
    TEST_1I("test_maxpool_2d", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2dc) {
    TEST_1I("test_maxpool_2d_ceil", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2dd) {
    TEST_1I("test_maxpool_2d_dilations", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_2dp) {
    TEST_1I("test_maxpool_2d_pads", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool_3d) {
    TEST_1I("test_maxpool_3d", x, y);
    return NULL;
}
UnitTest_fn_def(test_maxpool) {
    UnitTest_add(test_maxpool_1d);
    // UnitTest_add(test_maxpool_2d);
    // UnitTest_add(test_maxpool_2dc);
    // UnitTest_add(test_maxpool_2dd);  // Failed!
    // UnitTest_add(test_maxpool_2dp);
    // UnitTest_add(test_maxpool_3d);
    return NULL;
}

// ---------------------- Mul    ----------------------

UnitTest_fn_def(test_mul_dft) {
    TEST_2I("test_mul", x, y, z);
    return NULL;
}
UnitTest_fn_def(test_mul_bc) {
    TEST_2I("test_mul_bcast", x, y, z);
    return NULL;
}
UnitTest_fn_def(test_mul_ext) {
    TEST_2I("test_mul_example", x, y, z);
    return NULL;
}
UnitTest_fn_def(test_mul_u8) {
    TEST_2I("test_mul_uint8", x, y, z);
    return NULL;
}
UnitTest_fn_def(test_mul) {
    UnitTest_add(test_mul_dft);
    UnitTest_add(test_mul_bc);
    UnitTest_add(test_mul_ext);
    UnitTest_add(test_mul_u8);
    return NULL;
}


// ---------------------- Relu   ----------------------

UnitTest_fn_def(test_relu) {
    TEST_1I("test_relu", x, y);
    return NULL;
}

// ---------------------- Reshape ---------------------

UnitTest_fn_def(test_reshape_ar) {
    TEST_2I("test_reshape_allowzero_reordered", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_ed) {
    TEST_2I("test_reshape_extended_dims", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_nd) {
    TEST_2I("test_reshape_negative_dim", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_ned) {
    TEST_2I("test_reshape_negative_extended_dims", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_od) {
    TEST_2I("test_reshape_one_dim", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape_rd) {
    TEST_2I("test_reshape_reduced_dims", data, shape, reshaped);
    return NULL;
}
UnitTest_fn_def(test_reshape) {
    // UnitTest_add(test_reshape_ar); /** TODO: Segmentation fault */
    UnitTest_add(test_reshape_ed);
    UnitTest_add(test_reshape_nd);
    UnitTest_add(test_reshape_ned);
    UnitTest_add(test_reshape_od);
    UnitTest_add(test_reshape_rd);
    return NULL;
}

// ---------------------- Softmax ---------------------

UnitTest_fn_def(test_softmax_a0) {
    TEST_1I("test_softmax_axis_0", x, y);
    return NULL;
}
UnitTest_fn_def(test_softmax_a0e) {
    TEST_1I("test_softmax_axis_0_expanded", x, y);
    return NULL;
}
UnitTest_fn_def(test_softmax_a1) {
    TEST_1I("test_softmax_axis_1", x, y);
    return NULL;
}
UnitTest_fn_def(test_softmax_a1e) {
    TEST_1I("test_softmax_axis_1_expanded", x, y);
    return NULL;
}
UnitTest_fn_def(test_softmax_a2) {
    TEST_1I("test_softmax_axis_2", x, y);
    return NULL;
}
UnitTest_fn_def(test_softmax_a2e) {
    TEST_1I("test_softmax_axis_2_expanded", x, y);
    return NULL;
}
UnitTest_fn_def(test_softmax) {
    UnitTest_add(test_softmax_a0);
    // UnitTest_add(test_softmax_a0e);
    UnitTest_add(test_softmax_a1);
    // UnitTest_add(test_softmax_a1e);
    UnitTest_add(test_softmax_a2);
    // UnitTest_add(test_softmax_a2e);
    return NULL;
}

// ---------------------- Sum     ---------------------
UnitTest_fn_def(test_sum_1i) {
    TEST_1I("test_sum_one_input", data_0, result);
    return NULL;
}
UnitTest_fn_def(test_sum_2i) {
    TEST_2I("test_sum_two_input", data_0, data_1, result);
    return NULL;
}
UnitTest_fn_def(test_sum_3i) {
    TEST_3I("test_sum_three_input", data_0, data_1, data_2, result);
    return NULL;
}
UnitTest_fn_def(test_sum) {
    UnitTest_add(test_sum_1i);
    UnitTest_add(test_sum_2i);
    UnitTest_add(test_sum_3i);
    return NULL;
}

// ---------------------- Transpose -------------------

UnitTest_fn_def(test_transpose_dft) {
    TEST_1I("test_transpose_dft", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose_ap0) {
    TEST_1I("test_transpose_all_permutations_0", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose_ap1) {
    TEST_1I("test_transpose_all_permutations_1", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose_ap2) {
    TEST_1I("test_transpose_all_permutations_2", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose_ap3) {
    TEST_1I("test_transpose_all_permutations_3", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose_ap4) {
    TEST_1I("test_transpose_all_permutations_4", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose_ap5) {
    TEST_1I("test_transpose_all_permutations_5", data,  transposed);
    return NULL;
}
UnitTest_fn_def(test_transpose) {
    // UnitTest_add(test_transpose_dft);    // malloc(): invalid size
    UnitTest_add(test_transpose_ap0);
    UnitTest_add(test_transpose_ap1);
    UnitTest_add(test_transpose_ap2);
    UnitTest_add(test_transpose_ap3);
    UnitTest_add(test_transpose_ap4);
    UnitTest_add(test_transpose_ap5);
    return NULL;
}

// ---------------------- Exit   ----------------------

UnitTest_fn_def(test_node_exit) {
    serializer_free(sez);
    device_unreg("cpu");
    return NULL;
}

// ---------------------- All    ----------------------

UnitTest_fn_def(test_all) {
    UnitTest_add(test_node_init);

    UnitTest_add(test_abs);
    UnitTest_add(test_add);
    UnitTest_add(test_averagepool);
    // UnitTest_add(test_batchnorm);    // Failed
    UnitTest_add(test_cast);
    UnitTest_add(test_concat);
    UnitTest_add(test_conv);
    UnitTest_add(test_dropout);
    UnitTest_add(test_gemm);
    // UnitTest_add(test_globalaveragepool);    // Failed
    UnitTest_add(test_leakyrelu);
    UnitTest_add(test_matmul);
    UnitTest_add(test_maxpool);  // Failed
    UnitTest_add(test_mul);
    UnitTest_add(test_relu);
    UnitTest_add(test_reshape);
    UnitTest_add(test_softmax);
    UnitTest_add(test_sum);
    UnitTest_add(test_transpose);

    UnitTest_add(test_node_exit);
    return NULL;
}

UnitTest_run(test_all);