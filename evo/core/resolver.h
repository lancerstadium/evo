#include "../evo.h"
#include "../util/sys.h"

// ==================================================================================== //
//                                  resolver: default
// ==================================================================================== //

void op_Abs_dft(node_t*);
void op_Add_dft(node_t*);
void op_AveragePool_dft(node_t*);
void op_BatchNormalization_dft(node_t*);
void op_Cast_dft(node_t*);
void op_Concat_dft(node_t*);
void op_Constant_dft(node_t*);
void op_ConstantOfShape_dft(node_t*);
void op_Conv_dft(node_t*);
void op_Dropout_dft(node_t*);
void op_Equal_dft(node_t*);
void op_Expand_dft(node_t*);
void op_Gather_dft(node_t*);
void op_Gemm_dft(node_t*);
void op_InstanceNormalization_dft(node_t*);
void op_GlobalAveragePool_dft(node_t*);
void op_LeakyRelu_dft(node_t*);
void op_MatMul_dft(node_t*);
void op_MaxPool_dft(node_t*);
void op_Mul_dft(node_t*);
void op_Nop_dft(node_t*);
void op_Pad_dft(node_t*);
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
void op_Transpose_dft(node_t*);
void op_Unsqueeze_dft(node_t*);
void op_Where_dft(node_t*);