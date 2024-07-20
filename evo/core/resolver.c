#include "resolver.h"
#include "../util/log.h"

// ==================================================================================== //
//                                     operator
// ==================================================================================== //

static const char* op_name_tbl[OP_TYPE_LAST+1] = {
    [OP_TYPE_NOP] = "Nop",
    [OP_TYPE_ABS] = "Abs",
    [OP_TYPE_ACOS] = "Acos",
    [OP_TYPE_ACOSH] = "Acosh",
    [OP_TYPE_ADD] = "Add",
    [OP_TYPE_AND] = "And",
    [OP_TYPE_ARG_MAX] = "ArgMax",
    [OP_TYPE_ARG_MIN] = "ArgMin",
    [OP_TYPE_ASIN] = "Asin",
    [OP_TYPE_ASINH] = "Asinh",
    [OP_TYPE_ATAN] = "Atan",
    [OP_TYPE_ATANH] = "Atanh",
    [OP_TYPE_AVERAGE_POOL] = "AveragePool",
    [OP_TYPE_BATCH_NORMALIZATION] = "BatchNormalization",
    [OP_TYPE_BITSHIFT] = "BitShift",
    [OP_TYPE_CAST] = "Cast",
    [OP_TYPE_CEIL] = "Ceil",
    [OP_TYPE_CELU] = "Celu",
    [OP_TYPE_CLIP] = "Clip",
    [OP_TYPE_COMPRESS] = "Compress",
    [OP_TYPE_CONCAT] = "Concat",
    [OP_TYPE_CONCAT_FROM_SEQUENCE] = "ConcatFromSequence",
    [OP_TYPE_CONSTANT] = "Constant",
    [OP_TYPE_CONSTANT_OF_SHAPE] = "ConstantOfShape",
    [OP_TYPE_CONV] = "Conv",
    [OP_TYPE_CONV_INTERGER] = "ConvInteger",
    [OP_TYPE_CONV_TRANSPOSE] = "ConvTranspose",
    [OP_TYPE_COS] = "Cos",
    [OP_TYPE_COSH] = "Cosh",
    [OP_TYPE_CUM_SUM] = "CumSum",
    [OP_TYPE_DEPTH_TO_SPACE] = "DepthToSpace",
    [OP_TYPE_DEQUANTIZE_LINEAR] = "DequantizeLinear",
    [OP_TYPE_DET] = "Det",
    [OP_TYPE_DIV] = "Div",
    [OP_TYPE_DROPOUT] = "Dropout",
    [OP_TYPE_DYNAMIC_QUANTIZE_LINEAR] = "DynamicQuantizeLinear",
    [OP_TYPE_EINSUM] = "Einsum",
    [OP_TYPE_ELU] = "Elu",
    [OP_TYPE_EQUAL] = "Equal",
    [OP_TYPE_ERF] = "Erf",
    [OP_TYPE_EXP] = "Exp",
    [OP_TYPE_EXPAND] = "Expand",
    [OP_TYPE_EYELIKE] = "Eyelike",
    [OP_TYPE_FLATTEN] = "Flatten",
    [OP_TYPE_FLOOR] = "Floor",
    [OP_TYPE_GRU] = "Gru",
    [OP_TYPE_GATHER] = "Gather",
    [OP_TYPE_GATHER_ELEMENTS] = "GatherElements",
    [OP_TYPE_GATHER_ND] = "GatherNd",
    [OP_TYPE_GEMM] = "Gemm",
    [OP_TYPE_GLOBAL_AVERAGEPOOL] = "GlobalAveragePool",
    [OP_TYPE_GLOBAL_LP_POOL] = "GlobalLpPool",
    [OP_TYPE_GLOBAL_MAX_POOL] = "GlobalMaxPool",
    [OP_TYPE_GREATER] = "Greater",
    [OP_TYPE_GREATER_OR_EQUAL] = "GreaterOrEqual",
    [OP_TYPE_HARD_SIGMOID] = "HardSigmoid",
    [OP_TYPE_HARDMAX] = "Hardmax",
    [OP_TYPE_HARD_SWISH] = "HardSwish",
    [OP_TYPE_IDENTITY] = "Identity",
    [OP_TYPE_IF] = "If",
    [OP_TYPE_INSTANCE_NORMALIZATION] = "InstanceNormalization",
    [OP_TYPE_IS_INF] = "IsInf",
    [OP_TYPE_IS_NAN] = "IsNan",
    [OP_TYPE_LRN] = "Lrn",
    [OP_TYPE_LSTM] = "Lstm",
    [OP_TYPE_LEAKY_RELU] = "LeakyRelu",
    [OP_TYPE_LESS] = "Less",
    [OP_TYPE_LESS_OR_EQUAL] = "LessOrEqual",
    [OP_TYPE_LOG] = "Log",
    [OP_TYPE_LOG_SOFTMAX] = "LogSoftmax",
    [OP_TYPE_LOOP] = "Loop",
    [OP_TYPE_LP_NORMALIZATION] = "LpNormalization",
    [OP_TYPE_LP_POOL] = "LpPool",
    [OP_TYPE_MAT_MUL] = "MatMul",
    [OP_TYPE_MAT_MUL_INTERGER] = "MatMulInteger",
    [OP_TYPE_MAX] = "Max",
    [OP_TYPE_MAX_POOL] = "MaxPool",
    [OP_TYPE_MAX_ROI_POOL] = "MaxRoiPool",
    [OP_TYPE_MAX_UNPOOL] = "MaxUnpool",
    [OP_TYPE_MEAN] = "Mean",
    [OP_TYPE_MEAN_VARIANCE_NORMALIZATION] = "MeanVarianceNormalization",
    [OP_TYPE_MIN] = "Min",
    [OP_TYPE_MOD] = "Mod",
    [OP_TYPE_MUL] = "Mul",
    [OP_TYPE_MULTINOMIAL] = "Multinomial",
    [OP_TYPE_NEG] = "Neg",
    [OP_TYPE_NEGATIVE_LOG_LIKELIHOOD_LOSS] = "NegativeLogLikelihoodLoss",
    [OP_TYPE_NON_MAX_SUPPRESSION] = "NonMaxSuppression",
    [OP_TYPE_NON_ZERO] = "NonZero",
    [OP_TYPE_NOT] = "Not",
    [OP_TYPE_ONE_HOT] = "OneHot",
    [OP_TYPE_OR] = "Or",
    [OP_TYPE_PRELU] = "Prelu",
    [OP_TYPE_PAD] = "Pad",
    [OP_TYPE_POW] = "Pow",
    [OP_TYPE_QLINEAR_CONV] = "QlinearConv",
    [OP_TYPE_QLINEAR_MAT_MUL] = "QlinearMatMul",
    [OP_TYPE_QUANTIZE_LINEAR] = "QuantizeLinear",
    [OP_TYPE_RNN] = "Rnn",
    [OP_TYPE_RANDOM_NORMAL] = "RandomNormal",
    [OP_TYPE_RANDOM_NORMAL_LIKE] = "RandomNormalLike",
    [OP_TYPE_RANDOM_UNIFORM] = "RandomUniform",
    [OP_TYPE_RANDOM_UNIFORM_LIKE] = "RandomUniformLike",
    [OP_TYPE_RANGE] = "Range",
    [OP_TYPE_RECIPROCAL] = "Reciprocal",
    [OP_TYPE_REDUCE_L1] = "ReduceL1",
    [OP_TYPE_REDUCE_L2] = "ReduceL2",
    [OP_TYPE_REDUCE_LOG_SUM] = "ReduceLogSum",
    [OP_TYPE_REDUCE_LOG_SUM_EXP] = "ReduceLogSumExp",
    [OP_TYPE_REDUCE_MAX] = "ReduceMax",
    [OP_TYPE_REDUCE_MEAN] = "ReduceMean",
    [OP_TYPE_REDUCE_MIN] = "ReduceMin",
    [OP_TYPE_REDUCE_PROD] = "ReduceProd",
    [OP_TYPE_REDUCE_SUM] = "ReduceSum",
    [OP_TYPE_REDUCE_SUM_SQUARE] = "ReduceSumSquare",
    [OP_TYPE_RELU] = "Relu",
    [OP_TYPE_RESHAPE] = "Reshape",
    [OP_TYPE_RESIZE] = "Resize",
    [OP_TYPE_REVERSE_SEQUENCE] = "ReverseSequence",
    [OP_TYPE_ROI_ALIGN] = "RoiAlign",
    [OP_TYPE_ROUND] = "Round",
    [OP_TYPE_SCAN] = "Scan",
    [OP_TYPE_SCATTER] = "Scatter",
    [OP_TYPE_SCATTER_ELEMENTS] = "ScatterElements",
    [OP_TYPE_SCATTER_ND] = "ScatterNd",
    [OP_TYPE_SELU] = "Selu",
    [OP_TYPE_SEQUENCE_AT] = "SequenceAt",
    [OP_TYPE_SEQUENCE_CONSTRUCT] = "SequenceConstruct",
    [OP_TYPE_SEQUENCE_EMPTY] = "SequenceEmpty",
    [OP_TYPE_SEQUENCE_ERASE] = "SequenceErase",
    [OP_TYPE_SEQUENCE_INSERT] = "SequenceInsert",
    [OP_TYPE_SEQUENCE_LENGTH] = "SequenceLength",
    [OP_TYPE_SHAPE] = "Shape",
    [OP_TYPE_SHRINK] = "Shrink",
    [OP_TYPE_SIGMOID] = "Sigmoid",
    [OP_TYPE_SIGN] = "Sign",
    [OP_TYPE_SIN] = "Sin",
    [OP_TYPE_SINH] = "Sinh",
    [OP_TYPE_SIZE] = "Size",
    [OP_TYPE_SLICE] = "Slice",
    [OP_TYPE_SOFTMAX] = "Softmax",
    [OP_TYPE_SOFTMAX_CROSS_ENTROPY_LOSS] = "SoftmaxCrossEntropyLoss",
    [OP_TYPE_SOFTPLUS] = "Softplus",
    [OP_TYPE_SOFTSIGN] = "Softsign",
    [OP_TYPE_SPACE_TO_DEPTH] = "SpaceToDepth",
    [OP_TYPE_SPLIT] = "Split",
    [OP_TYPE_SPLIT_TO_SEQUENCE] = "SplitToSequence",
    [OP_TYPE_SQRT] = "Sqrt",
    [OP_TYPE_SQUEEZE] = "Squeeze",
    [OP_TYPE_STRING_NORMALIZER] = "StringNormalizer",
    [OP_TYPE_SUB] = "Sub",
    [OP_TYPE_SUM] = "Sum",
    [OP_TYPE_TAN] = "Tan",
    [OP_TYPE_TANH] = "Tanh",
    [OP_TYPE_TF_IDF_VECTORIZER] = "TfIdfVectorizer",
    [OP_TYPE_THRESHOLDED_RELU] = "ThresholdedRelu",
    [OP_TYPE_TILE] = "Tile",
    [OP_TYPE_TOP_K] = "TopK",
    [OP_TYPE_TRANSPOSE] = "Transpose",
    [OP_TYPE_TRILU] = "Trilu",
    [OP_TYPE_UNIQUE] = "Unique",
    [OP_TYPE_UNSQUEEZE] = "Unsqueeze",
    [OP_TYPE_UPSAMPLE] = "Upsample",
    [OP_TYPE_WHERE] = "Where",
    [OP_TYPE_XOR] = "Xor",
    [OP_TYPE_LAST] = NULL
};

const char* op_name(op_type_t type) {
    if((int)type >= 0 && type < OP_TYPE_LAST)
        return op_name_tbl[type];
    return NULL;
}

// ==================================================================================== //
//                                  resolver: default
// ==================================================================================== //

static void* resolver_init_dft() {
    return NULL;
}

static void resolver_release_dft(void* rctx) {
}

#define OP_REG(T, S, R) [OP_TYPE_##T] = {.type = OP_TYPE_##T, .run = op_##S##_##R}
#define OP_REG_DFT(T, S) OP_REG(T, S, dft)

static resolver_t default_resolver = {
    .name = "default",
    .init = resolver_init_dft,
    .release = resolver_release_dft,

    .op_tbl = (op_t[]){
        OP_REG_DFT(NOP                  , Nop                   ),
        OP_REG_DFT(ABS                  , Abs                   ),
        OP_REG_DFT(ADD                  , Add                   ),
        OP_REG_DFT(AVERAGE_POOL         , AveragePool           ),
        OP_REG_DFT(BATCH_NORMALIZATION  , BatchNormalization    ),
        OP_REG_DFT(CONCAT               , Concat                ),
        OP_REG_DFT(CONV                 , Conv                  ),
        OP_REG_DFT(DROPOUT              , Dropout               ),
        OP_REG_DFT(GLOBAL_AVERAGEPOOL   , GlobalAveragePool     ),
        OP_REG_DFT(LEAKY_RELU           , LeakyRelu             ),
        OP_REG_DFT(MAT_MUL              , MatMul                ),
        OP_REG_DFT(MAX_POOL             , MaxPool               ),
        OP_REG_DFT(MUL                  , Mul                   ),
        OP_REG_DFT(RELU                 , Relu                  ),
        OP_REG_DFT(RESHAPE              , Reshape               ),
        [OP_TYPE_LAST] = {0}
    }
};

#undef OP_REG_DFT
#undef OP_REG


// ==================================================================================== //
//                                  resolver API
// ==================================================================================== //

resolver_t* resolver_get_default() {
    return &default_resolver;
}