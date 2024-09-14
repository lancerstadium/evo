#include <evo.h>
#include <evo/util/sys.h>

// ==================================================================================== //
//                                  resolver: default
// ==================================================================================== //

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void op_Abs_dft(node_t*);
void op_Add_dft(node_t*);
void op_ArgMax_dft(node_t*);
void op_AveragePool_dft(node_t*);
void op_BatchNormalization_dft(node_t*);
void op_Cast_dft(node_t*);
void op_Concat_dft(node_t*);
void op_Constant_dft(node_t*);
void op_ConstantOfShape_dft(node_t*);
void op_Conv_dft(node_t*);
void op_Div_dft(node_t*);
void op_Dropout_dft(node_t*);
void op_Equal_dft(node_t*);
void op_Expand_dft(node_t*);
void op_Flatten_dft(node_t*);
void op_Gather_dft(node_t*);
void op_Gemm_dft(node_t*);
void op_InstanceNormalization_dft(node_t*);
void op_GlobalAveragePool_dft(node_t*);
void op_LeakyRelu_dft(node_t*);
void op_Log_dft(node_t*);
void op_MatMul_dft(node_t*);
void op_MaxPool_dft(node_t*);
void op_Mul_dft(node_t*);
void op_Neg_dft(node_t*);
void op_Nop_dft(node_t*);
void op_Pad_dft(node_t*);
void op_RandomUniform_dft(node_t*);
void op_RandomUniformLike_dft(node_t*);
void op_Range_dft(node_t*);
void op_Relu_dft(node_t*);
void op_Reshape_dft(node_t*);
void op_Resize_dft(node_t*);
void op_ScatterNd_dft(node_t*);
void op_Shape_dft(node_t*);
void op_Slice_dft(node_t*);
void op_Softmax_dft(node_t*);
void op_Sub_dft(node_t*);
void op_Sum_dft(node_t*);
void op_Tanh_dft(node_t*);
void op_Squeeze_dft(node_t*);
void op_Transpose_dft(node_t*);
void op_Unsqueeze_dft(node_t*);
void op_Upsample_dft(node_t*);
void op_Where_dft(node_t*);


#ifdef __cplusplus
}
#endif  // __cplusplus