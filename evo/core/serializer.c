#include "../evo.h"
#include "../util/log.h"
#include "../util/math.h"
#include "../util/onnx.proto3.pb-c.h"
#include "../util/sys.h"
#include <string.h>
#include <stdio.h>

// ==================================================================================== //
//                                      dummy
// ==================================================================================== //

EVO_UNUSED static int reshape_dummy(node_t *n) {
    return 1;
}

EVO_UNUSED static void operator_dummy(node_t *n) {
    LOG_WARN("\033[45;37mUnsupported opset\033[0m => %s-%d (%s)\r\n", ((Onnx__NodeProto *)(n->node_proto))->op_type, n->opset, (strlen(((Onnx__NodeProto *)(n->node_proto))->domain) > 0) ? ((Onnx__NodeProto *)(n->node_proto))->domain : "ai.onnx");
}

// ==================================================================================== //
//                                      onnx
// ==================================================================================== //

context_t *load_model_onnx(struct serializer *sez, const char *path);
void unload_onnx(context_t *ctx);
tensor_t *load_tensor_onnx(const char *path);
graph_t *load_graph_onnx(context_t *ctx);
context_t *load_onnx(struct serializer *s, const void *buf, int len);

EVO_UNUSED static op_type_t op_map_onnx(char *op_ty) {
    if (!op_ty) return OP_TYPE_NOP;
    LOG_INFO("++ op ty: %-12s, hash: 0x%08x\n", op_ty, shash(op_ty));
    switch (shash(op_ty)) {
        case 0x0b87d47b:
            return OP_TYPE_ABS; /* "Abs" */
        case 0x7c82680b:
            return OP_TYPE_ACOS; /* "Acos" */
        case 0x0ccf69d3:
            return OP_TYPE_ACOSH; /* "Acosh" */
        case 0x0b87d4ae:
            return OP_TYPE_ADD; /* "Add" */
        case 0x0b87d5f8:
            return OP_TYPE_AND; /* "And" */
        case 0xa7c70ea5:
            return OP_TYPE_ARG_MAX; /* "ArgMax" */
        case 0xa7c70fa3:
            return OP_TYPE_ARG_MIN; /* "ArgMin" */
        case 0x7c82ab50:
            return OP_TYPE_ASIN; /* "Asin" */
        case 0x0cd815b8:
            return OP_TYPE_ASINH; /* "Asinh" */
        case 0x7c82ae89:
            return OP_TYPE_ATAN; /* "Atan" */
        case 0x0cd88011:
            return OP_TYPE_ATANH; /* "Atanh" */
        case 0xf1a1e23a:
            return OP_TYPE_AVERAGE_POOL; /* "AveragePool" */
        case 0x2d3b46ee:
            return OP_TYPE_BATCH_NORMALIZATION; /* "BatchNormalization" */
        case 0x0bfe45a2:
            return OP_TYPE_BITSHIFT; /* "BitShift" */
        case 0x7c8378d0:
            return OP_TYPE_CAST; /* "Cast" */
        case 0x7c838882:
            return OP_TYPE_CEIL; /* "Ceil" */
        case 0x7c83a64d:
            return OP_TYPE_CLIP; /* "Clip" */
        case 0xb7db9db1:
            return OP_TYPE_COMPRESS; /* "Compress" */
        case 0xac3f4a9d:
            return OP_TYPE_CONCAT; /* "Concat" */
        case 0x5053caca:
            return OP_TYPE_CONCAT_FROM_SEQUENCE; /* "ConcatFromSequence" */
        case 0xba6816ef:
            return OP_TYPE_CONSTANT; /* "Constant" */
        case 0xe468a875:
            return OP_TYPE_CONSTANT_OF_SHAPE; /* "ConstantOfShape" */
        case 0x7c83b3bb:
            return OP_TYPE_CONV; /* "Conv" */
        case 0x8371dbe9:
            return OP_TYPE_CONV_INTERGER; /* "ConvInteger" */
        case 0x3903c4ba:
            return OP_TYPE_CONV_TRANSPOSE; /* "ConvTranspose" */
        case 0x0b87deaa:
            return OP_TYPE_COS; /* "Cos" */
        case 0x7c83b452:
            return OP_TYPE_COSH; /* "Cosh" */
        case 0xacab0fbf:
            return OP_TYPE_CUM_SUM; /* "CumSum" */
        case 0xc9c1d669:
            return OP_TYPE_DEPTH_TO_SPACE; /* "DepthToSpace" */
        case 0xf9cc985a:
            return OP_TYPE_DEQUANTIZE_LINEAR; /* "DequantizeLinear" */
        case 0x0b87e1a2:
            return OP_TYPE_DET; /* "Det" */
        case 0x0b87e228:
            return OP_TYPE_DIV; /* "Div" */
        case 0x883bca72:
            return OP_TYPE_DROPOUT; /* "Dropout" */
        case 0xb07d4f76:
            return OP_TYPE_EINSUM; /* "Einsum" */
        case 0x0b87e6cb:
            return OP_TYPE_ELU; /* "Elu" */
        case 0x0d1f905d:
            return OP_TYPE_EQUAL; /* "Equal" */
        case 0x0b87e782:
            return OP_TYPE_ERF; /* "Erf" */
        case 0x0b87e852:
            return OP_TYPE_EXP; /* "Exp" */
        case 0xb18d8a45:
            return OP_TYPE_EXPAND; /* "Expand" */
        case 0xe4c1560d:
            return OP_TYPE_EYELIKE; /* "EyeLike" */
        case 0x13363dd3:
            return OP_TYPE_FLATTEN; /* "Flatten" */
        case 0x0d2ed347:
            return OP_TYPE_FLOOR; /* "Floor" */
        case 0x0b87ebd3:
            return OP_TYPE_GRU; /* "GRU" */
        case 0xb499f620:
            return OP_TYPE_GATHER; /* "Gather" */
        case 0x7c94d43d:
            return OP_TYPE_GATHER_ELEMENTS; /* "GatherElements" */
        case 0x42f00872:
            return OP_TYPE_GATHER_ND; /* "GatherND" */
        case 0x7c85ba8b:
            return OP_TYPE_GEMM; /* "Gemm" */
        case 0x9289c84b:
            return OP_TYPE_GLOBAL_AVERAGEPOOL; /* "GlobalAveragePool" */
        case 0x3f5a29ac:
            return OP_TYPE_GLOBAL_LP_POOL; /* "GlobalLpPool" */
        case 0x575f0fb6:
            return OP_TYPE_GLOBAL_MAX_POOL; /* "GlobalMaxPool" */
        case 0x6e6d652f:
            return OP_TYPE_GREATER; /* "Greater" */
        case 0x10341df0:
            return OP_TYPE_HARD_SIGMOID; /* "HardSigmoid" */
        case 0x94acb4aa:
            return OP_TYPE_HARDMAX; /* "Hardmax" */
        case 0xdfd9b28f:
            return OP_TYPE_IDENTITY; /* "Identity" */
        case 0x00597414:
            return OP_TYPE_IF; /* "If" */
        case 0xfb0902c1:
            return OP_TYPE_INSTANCE_NORMALIZATION; /* "InstanceNormalization" */
        case 0x0d68519e:
            return OP_TYPE_IS_INF; /* "IsInf" */
        case 0x0d68651e:
            return OP_TYPE_IS_NAN; /* "IsNaN" */
        case 0x0b880111:
            return OP_TYPE_LRN; /* "LRN" */
        case 0x7c882885:
            return OP_TYPE_LSTM; /* "LSTM" */
        case 0xea2c5c33:
            return OP_TYPE_LEAKY_RELU; /* "LeakyRelu" */
        case 0x7c88793c:
            return OP_TYPE_LESS; /* "Less" */
        case 0x0b8804e7:
            return OP_TYPE_LOG; /* "Log" */
        case 0x7c88a33f:
            return OP_TYPE_LOOP; /* "Loop" */
        case 0x07f77ce8:
            return OP_TYPE_LP_NORMALIZATION; /* "LpNormalization" */
        case 0xc13f923b:
            return OP_TYPE_LP_POOL; /* "LpPool" */
        case 0xc2987915:
            return OP_TYPE_MAT_MUL; /* "MatMul" */
        case 0x62fbd803:
            return OP_TYPE_MAT_MUL_INTERGER; /* "MatMulInteger" */
        case 0x0b88076b:
            return OP_TYPE_MAX; /* "Max" */
        case 0x15f18a25:
            return OP_TYPE_MAX_POOL; /* "MaxPool" */
        case 0x018c06cf:
            return OP_TYPE_MAX_ROI_POOL; /* "MaxRoiPool" */
        case 0x641501e8:
            return OP_TYPE_MAX_UNPOOL; /* "MaxUnpool" */
        case 0x7c890346:
            return OP_TYPE_MEAN; /* "Mean" */
        case 0x0b880869:
            return OP_TYPE_MIN; /* "Min" */
        case 0x0b880925:
            return OP_TYPE_MOD; /* "Mod" */
        case 0x0b8809f3:
            return OP_TYPE_MUL; /* "Mul" */
        case 0xaec55410:
            return OP_TYPE_MULTINOMIAL; /* "Multinomial" */
        case 0x0b880c1f:
            return OP_TYPE_NEG; /* "Neg" */
        case 0x254e25a1:
            return OP_TYPE_NON_MAX_SUPPRESSION; /* "NonMaxSuppression" */
        case 0x82e45c50:
            return OP_TYPE_NON_ZERO; /* "NonZero" */
        case 0x0b880d76:
            return OP_TYPE_NOT; /* "Not" */
        case 0xc825b932:
            return OP_TYPE_ONE_HOT; /* "OneHot" */
        case 0x005974e6:
            return OP_TYPE_OR; /* "Or" */
        case 0x0dd55b8d:
            return OP_TYPE_PRELU; /* "PRelu" */
        case 0x0b88141a:
            return OP_TYPE_PAD; /* "Pad" */
        case 0x0b8815fb:
            return OP_TYPE_POW; /* "Pow" */
        case 0xe569f427:
            return OP_TYPE_QLINEAR_CONV; /* "QLinearConv" */
        case 0xfe108481:
            return OP_TYPE_QLINEAR_MAT_MUL; /* "QLinearMatMul" */
        case 0x37138211:
            return OP_TYPE_QUANTIZE_LINEAR; /* "QuantizeLinear" */
        case 0x0b881a13:
            return OP_TYPE_RNN; /* "RNN" */
        case 0xc100684f:
            return OP_TYPE_RANDOM_NORMAL; /* "RandomNormal" */
        case 0xa0b57174:
            return OP_TYPE_RANDOM_NORMAL_LIKE; /* "RandomNormalLike" */
        case 0xf8e97c66:
            return OP_TYPE_RANDOM_UNIFORM; /* "RandomUniform" */
        case 0x10a8b90b:
            return OP_TYPE_RANDOM_UNIFORM_LIKE; /* "RandomUniformLike" */
        case 0x73d06f69:
            return OP_TYPE_RECIPROCAL; /* "Reciprocal" */
        case 0x7944853a:
            return OP_TYPE_REDUCE_L1; /* "ReduceL1" */
        case 0x7944853b:
            return OP_TYPE_REDUCE_L2; /* "ReduceL2" */
        case 0xeab46d14:
            return OP_TYPE_REDUCE_LOG_SUM; /* "ReduceLogSum" */
        case 0x9a057a01:
            return OP_TYPE_REDUCE_LOG_SUM_EXP; /* "ReduceLogSumExp" */
        case 0xa1d53763:
            return OP_TYPE_REDUCE_MAX; /* "ReduceMax" */
        case 0xdc7c323e:
            return OP_TYPE_REDUCE_MEAN; /* "ReduceMean" */
        case 0xa1d53861:
            return OP_TYPE_REDUCE_MIN; /* "ReduceMin" */
        case 0xdc7e1072:
            return OP_TYPE_REDUCE_PROD; /* "ReduceProd" */
        case 0xa1d55372:
            return OP_TYPE_REDUCE_SUM; /* "ReduceSum" */
        case 0x20917223:
            return OP_TYPE_REDUCE_SUM_SQUARE; /* "ReduceSumSquare" */
        case 0x7c8bc29d:
            return OP_TYPE_RELU; /* "Relu" */
        case 0x9fdbcf8d:
            return OP_TYPE_RESHAPE; /* "Reshape" */
        case 0xce8a9197:
            return OP_TYPE_RESIZE; /* "Resize" */
        case 0x5d77301a:
            return OP_TYPE_REVERSE_SEQUENCE; /* "ReverseSequence" */
        case 0x830cb9da:
            return OP_TYPE_ROI_ALIGN; /* "RoiAlign" */
        case 0x0e09b7cd:
            return OP_TYPE_ROUND; /* "Round" */
        case 0x7c8c450a:
            return OP_TYPE_SCAN; /* "Scan" */
        case 0xe6ece5fb:
            return OP_TYPE_SCATTER; /* "Scatter" */
        case 0xb4db6f18:
            return OP_TYPE_SCATTER_ELEMENTS; /* "ScatterElements" */
        case 0x55be5b0d:
            return OP_TYPE_SCATTER_ND; /* "ScatterND" */
        case 0x7c8c4efe:
            return OP_TYPE_SELU; /* "Selu" */
        case 0xe537ccd3:
            return OP_TYPE_SEQUENCE_AT; /* "SequenceAt" */
        case 0xa52772e3:
            return OP_TYPE_SEQUENCE_CONSTRUCT; /* "SequenceConstruct" */
        case 0x5e6e772d:
            return OP_TYPE_SEQUENCE_EMPTY; /* "SequenceEmpty" */
        case 0x5e70f50e:
            return OP_TYPE_SEQUENCE_ERASE; /* "SequenceErase" */
        case 0x35a57cb3:
            return OP_TYPE_SEQUENCE_INSERT; /* "SequenceInsert" */
        case 0x3bff64e0:
            return OP_TYPE_SEQUENCE_LENGTH; /* "SequenceLength" */
        case 0x0e17a4d6:
            return OP_TYPE_SHAPE; /* "Shape" */
        case 0xd11575d4:
            return OP_TYPE_SHRINK; /* "Shrink" */
        case 0xf5548151:
            return OP_TYPE_SIGMOID; /* "Sigmoid" */
        case 0x7c8c5f56:
            return OP_TYPE_SIGN; /* "Sign" */
        case 0x0b8821ef:
            return OP_TYPE_SIN; /* "Sin" */
        case 0x7c8c6037:
            return OP_TYPE_SINH; /* "Sinh" */
        case 0x7c8c61c0:
            return OP_TYPE_SIZE; /* "Size" */
        case 0x0e19f6b5:
            return OP_TYPE_SLICE; /* "Slice" */
        case 0x6bec36a5:
            return OP_TYPE_SOFTPLUS; /* "Softplus" */
        case 0x6bedcd32:
            return OP_TYPE_SOFTSIGN; /* "Softsign" */
        case 0xa4436289:
            return OP_TYPE_SPACE_TO_DEPTH; /* "SpaceToDepth" */
        case 0x0e1c35d1:
            return OP_TYPE_SPLIT; /* "Split" */
        case 0x50e66fcd:
            return OP_TYPE_SPLIT_TO_SEQUENCE; /* "SplitToSequence" */
        case 0x7c8c82cf:
            return OP_TYPE_SQRT; /* "Sqrt" */
        case 0x08f69207:
            return OP_TYPE_SQUEEZE; /* "Squeeze" */
        case 0xf404645f:
            return OP_TYPE_STRING_NORMALIZER; /* "StringNormalizer" */
        case 0x0b88236f:
            return OP_TYPE_SUB; /* "Sub" */
        case 0x0b88237a:
            return OP_TYPE_SUM; /* "Sum" */
        case 0x0b882528:
            return OP_TYPE_TAN; /* "Tan" */
        case 0x7c8cca90:
            return OP_TYPE_TANH; /* "Tanh" */
        case 0x46fbf3df:
            return OP_TYPE_TF_IDF_VECTORIZER; /* "TfIdfVectorizer" */
        case 0xa646ea33:
            return OP_TYPE_THRESHOLDED_RELU; /* "ThresholdedRelu" */
        case 0x7c8cec53:
            return OP_TYPE_TILE; /* "Tile" */
        case 0x7c8d0643:
            return OP_TYPE_TOP_K; /* "TopK" */
        case 0x940b3944:
            return OP_TYPE_TRANSPOSE; /* "Transpose" */
        case 0xd6278d9c:
            return OP_TYPE_UNIQUE; /* "Unique" */
        case 0xc836156a:
            return OP_TYPE_UNSQUEEZE; /* "Unsqueeze" */
        case 0xae63c66c:
            return OP_TYPE_UPSAMPLE; /* "Upsample" */
        case 0x0e601820:
            return OP_TYPE_WHERE; /* "Where" */
        case 0x0b8837fe:
            return OP_TYPE_XOR; /* "Xor" */
        case 0x7c8388ee:
            return OP_TYPE_CELU; /* "Celu" */
        case 0x718dbc56:
            return OP_TYPE_DYNAMIC_QUANTIZE_LINEAR; /* "DynamicQuantizeLinear" */
        case 0x7b2541c8:
            return OP_TYPE_GREATER_OR_EQUAL; /* "GreaterOrEqual" */
        case 0x60d9a535:
            return OP_TYPE_LESS_OR_EQUAL; /* "LessOrEqual" */
        case 0xf8c82769:
            return OP_TYPE_LOG_SOFTMAX; /* "LogSoftmax" */
        case 0xbb8f2396:
            return OP_TYPE_MEAN_VARIANCE_NORMALIZATION; /* "MeanVarianceNormalization" */
        case 0x6ed111df:
            return OP_TYPE_NEGATIVE_LOG_LIKELIHOOD_LOSS; /* "NegativeLogLikelihoodLoss" */
        case 0x0e01ebd2:
            return OP_TYPE_RANGE; /* "Range" */
        case 0x034529c7:
            return OP_TYPE_SOFTMAX; /* "Softmax" */
        case 0x522154a3:
            return OP_TYPE_SOFTMAX_CROSS_ENTROPY_LOSS; /* "SoftmaxCrossEntropyLoss" */
        default:
            return OP_TYPE_NOP;
    }
}

context_t *load_model_onnx(struct serializer *sez, const char *path) {
    context_t *ctx = NULL;
    FILE *fp;
    uint32_t len;
    unsigned int i;
    void *buf;
    fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        len = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        if (len > 0) {
            buf = sys_malloc(len);
            if (buf) {
                for (i = 0; i < len; i += fread(buf + i, 1, len - i, fp));
                ctx = load_onnx(sez, buf, len);
                sys_free(buf);
            }
        }
        fclose(fp);
    }
    return ctx;
}

void unload_onnx(context_t *ctx) {
    if (ctx && ctx->model) {
        onnx__model_proto__free_unpacked(ctx->model, NULL);
        ctx->model_size = 0;
    }
}

EVO_UNUSED static tensor_t *tensor_from_value_info(Onnx__ValueInfoProto *v) {
    tensor_t *t;
    tensor_type_t type;
    int *dims = NULL;
    int ndim;
    int i;

    if (!v || !v->name)
        return NULL;

    switch (v->type->value_case) {
        case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
            type = (tensor_type_t)v->type->tensor_type->elem_type;
            ndim = v->type->tensor_type->shape->n_dim;
            if (ndim > 0) {
                dims = sys_malloc(sizeof(int) * ndim);
                if (dims) {
                    for (i = 0; i < ndim; i++) {
                        switch (v->type->tensor_type->shape->dim[i]->value_case) {
                            case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
                                dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
                                break;
                            case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
                                if (strcmp(v->type->tensor_type->shape->dim[i]->dim_param, "batch_size") == 0)
                                    dims[i] = 1;
                                else
                                    dims[i] = 1;
                                break;
                            default:
                                dims[i] = 1;
                                break;
                        }
                    }
                }
            }
            t = tensor_new(v->name, type);
            tensor_reshape(t, ndim, dims);
            if (dims)
                sys_free(dims);
            break;
        case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
            t = NULL;
            break;
        case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
            t = NULL;
            break;
        default:
            t = NULL;
            break;
    }
    return t;
}

EVO_UNUSED static void tensor_copy_proto(tensor_t *t, Onnx__TensorProto *o) {
    size_t n, i;
    int sz;

    if (t && o) {
        if (t->type == o->data_type) {
            sz = tensor_type_sizeof(t->type);
            if (sz > 0) {
                if ((o->raw_data.len > 0) && o->raw_data.data) {
                    switch (o->data_type) {
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: {
                            float *p = (float *)t->datas;
                            uint32_t *q = (uint32_t *)o->raw_data.data;
                            union {
                                uint32_t u;
                                float f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++) {
                                    v.u = le32_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: {
                            uint8_t *p = (uint8_t *)t->datas;
                            uint8_t *q = (uint8_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len);
                                memcpy(p, q, n);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: {
                            int8_t *p = (int8_t *)t->datas;
                            int8_t *q = (int8_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len);
                                memcpy(p, q, n);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16: {
                            uint16_t *p = (uint16_t *)t->datas;
                            uint16_t *q = (uint16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT16: {
                            int16_t *p = (int16_t *)t->datas;
                            int16_t *q = (int16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: {
                            int32_t *p = (int32_t *)t->datas;
                            int32_t *q = (int32_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le32_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: {
                            int64_t *p = (int64_t *)t->datas;
                            int64_t *q = (int64_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le64_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL: {
                            uint8_t *p = (uint8_t *)t->datas;
                            uint8_t *q = (uint8_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len);
                                memcpy(p, q, n);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16: {
                            uint16_t *p = (uint16_t *)t->datas;
                            uint16_t *q = (uint16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: {
                            double *p = (double *)t->datas;
                            uint64_t *q = (uint64_t *)o->raw_data.data;
                            union {
                                uint64_t u;
                                double f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++) {
                                    v.u = le64_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32: {
                            uint32_t *p = (uint32_t *)t->datas;
                            uint32_t *q = (uint32_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le32_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64: {
                            uint64_t *p = (uint64_t *)t->datas;
                            uint64_t *q = (uint64_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le64_to_cpu(q[i]);
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64: {
                            float *p = (float *)t->datas;
                            uint32_t *q = (uint32_t *)o->raw_data.data;
                            union {
                                uint32_t u;
                                float f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz) * 2;
                                for (i = 0; i < n; i++) {
                                    v.u = le32_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128: {
                            double *p = (double *)t->datas;
                            uint64_t *q = (uint64_t *)o->raw_data.data;
                            union {
                                uint64_t u;
                                double f;
                            } v;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz) * 2;
                                for (i = 0; i < n; i++) {
                                    v.u = le64_to_cpu(q[i]);
                                    p[i] = v.f;
                                }
                            }
                        } break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16: {
                            uint16_t *p = (uint16_t *)t->datas;
                            uint16_t *q = (uint16_t *)o->raw_data.data;
                            if (t->ndata > 0) {
                                n = MIN(t->ndata, (size_t)o->raw_data.len / sz);
                                for (i = 0; i < n; i++)
                                    p[i] = le16_to_cpu(q[i]);
                            }
                        } break;
                        default:
                            break;
                    }
                } else {
                    switch (o->data_type) {
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
                            n = MIN(t->ndata, (size_t)o->n_float_data);
                            if ((n > 0) && t->datas && o->float_data)
                                memcpy(t->datas, o->float_data, sizeof(float) * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
                            // TODO
                            n = MIN(t->ndata, (size_t)o->n_int32_data);
                            if ((n > 0) && t->datas && o->int32_data)
                                memcpy(t->datas, o->int32_data, sz * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
                            n = MIN(t->ndata, (size_t)o->n_string_data);
                            if ((n > 0) && t->datas && o->string_data) {
                                char **str = (char **)t->datas;
                                for (i = 0; i < t->ndata; i++) {
                                    if (str[i]) {
                                        sys_free(str[i]);
                                        str[i] = NULL;
                                    }
                                }
                                for (i = 0; i < n; i++) {
                                    str[i] = sys_malloc(o->string_data[i].len + 1);
                                    if (str[i]) {
                                        str[i][o->string_data[i].len] = 0;
                                        memcpy(str[i], o->string_data[i].data, o->string_data[i].len);
                                    }
                                }
                            }
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
                            n = MIN(t->ndata, (size_t)o->n_int64_data);
                            if ((n > 0) && t->datas && o->int64_data)
                                memcpy(t->datas, o->int64_data, sizeof(int64_t) * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
                            n = MIN(t->ndata, (size_t)o->n_double_data);
                            if ((n > 0) && t->datas && o->double_data)
                                memcpy(t->datas, o->double_data, sizeof(double) * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
                        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
                            // TODO
                            n = MIN(t->ndata, (size_t)o->n_uint64_data);
                            if ((n > 0) && t->datas && o->uint64_data)
                                memcpy(t->datas, o->uint64_data, sz * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
                            n = MIN(t->ndata, (size_t)(o->n_float_data / 2));
                            if ((n > 0) && t->datas && o->float_data)
                                memcpy(t->datas, o->float_data, sizeof(float) * 2 * n);
                            break;
                        case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
                            n = MIN(t->ndata, (size_t)(o->n_double_data / 2));
                            if ((n > 0) && t->datas && o->double_data)
                                memcpy(t->datas, o->double_data, sizeof(double) * 2 * n);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

tensor_t *load_tensor_onnx(const char *path) {
    tensor_t *t = NULL;
    Onnx__TensorProto *pb;
    FILE *fp;
    void *buf;
    size_t l, len;
    int *dims = NULL;
    int ndim = 0;
    int i;

    fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        l = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        if (l > 0) {
            buf = sys_malloc(l);
            if (buf) {
                for (len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
                pb = onnx__tensor_proto__unpack(NULL, len, buf);
                sys_free(buf);
                if (pb) {
                    if (pb->n_dims > 0) {
                        dims = (int *)sys_malloc(sizeof(int) * pb->n_dims);
                        if (dims) {
                            for (i = 0; i < pb->n_dims; i++)
                                dims[i] = pb->dims[i];
                            ndim = pb->n_dims;
                        }
                    }
                    t = tensor_new(pb->name, (tensor_type_t)pb->data_type);
                    tensor_reshape(t, ndim, dims);
                    if ((ndim > 0) && dims)
                        sys_free(dims);
                    tensor_copy_proto(t, pb);
                    onnx__tensor_proto__free_unpacked(pb, NULL);
                }
            }
        }
        fclose(fp);
    }
    return t;
}

graph_t *load_graph_onnx(context_t *ctx) {
    if (!ctx || !ctx->model) {
        return NULL;
    }
    EVO_UNUSED Onnx__GraphProto *graph = ((Onnx__ModelProto *)(ctx->model))->graph;
    EVO_UNUSED graph_t *g;
    EVO_UNUSED node_t *n;
    EVO_UNUSED tensor_t *t;
    EVO_UNUSED Onnx__TensorProto *o;
    EVO_UNUSED Onnx__ValueInfoProto *v;
    EVO_UNUSED char *p, *domain;
    EVO_UNUSED char *name;
    EVO_UNUSED int i, j, k, l;

    if (!graph)
        return NULL;

    g = graph_new(ctx);
    if (!g)
        return NULL;

    g->nnode = graph->n_node;
    g->nodes = (node_t **)sys_malloc(sizeof(node_t *) * g->nnode);
    if (!g->nodes) {
        sys_free(g);
        return NULL;
    }

    // deal with input
    for (i = 0; i < graph->n_input; i++) {
        v = graph->input[i];
        if (!context_get_tensor(ctx, v->name)) {
            t = tensor_from_value_info(v);
            if (t) {
                for (j = 0; j < graph->n_initializer; j++) {
                    if (strcmp(graph->initializer[j]->name, t->name) == 0) {
                        tensor_copy_proto(t, graph->initializer[j]);
                        break;
                    }
                }
                hashmap_set(ctx->tensor_map, hashmap_str_lit(t->name), (uintptr_t)t);
            }
        }
    }
    // deal with output
    for (i = 0; i < graph->n_output; i++) {
        v = graph->output[i];
        if (!context_get_tensor(ctx, v->name)) {
            t = tensor_from_value_info(v);
            if (t) {
                hashmap_set(ctx->tensor_map, hashmap_str_lit(t->name), (uintptr_t)t);
            }
        }
    }
    // deal with value info
    for (i = 0; i < graph->n_value_info; i++) {
        v = graph->value_info[i];
        if (!context_get_tensor(ctx, v->name)) {
            t = tensor_from_value_info(v);
            if (t) {
                hashmap_set(ctx->tensor_map, hashmap_str_lit(t->name), (uintptr_t)t);
            }
        }
    }
    // deal with node output
    for (i = 0; i < graph->n_node; i++) {
        for (j = 0; j < graph->node[i]->n_output; j++) {
            name = graph->node[i]->output[j];
            if (!context_get_tensor(ctx, name)) {
                t = tensor_new(name, TENSOR_TYPE_UNDEFINED);
                if (t) hashmap_set(ctx->tensor_map, hashmap_str_lit(name), (uintptr_t)t);
            }
        }
    }

    // deal with node input
    for (i = 0; i < graph->n_node; i++) {
        for (j = 0; j < graph->node[i]->n_input; j++) {
            name = graph->node[i]->input[j];
            if (!context_get_tensor(ctx, name)) {
                for (k = 0; k < graph->n_initializer; k++) {
                    if (strcmp(graph->initializer[k]->name, name) == 0) {
                        o = graph->initializer[k];
                        if (o) {
                            int ndim = o->n_dims;
                            int dims[ndim];
                            for (l = 0; l < ndim; l++) {
                                dims[l] = o->dims[l];
                            }
                            t = tensor_new(name, TENSOR_TYPE_UNDEFINED);
                            if (t) {
                                tensor_reshape(t, ndim, dims);
                                tensor_copy_proto(t, o);
                                hashmap_set(ctx->tensor_map, hashmap_str_lit(name), (uintptr_t)t);
                            }
                            break;
                        }
                    }
                }
                if (!context_get_tensor(ctx, name)) {
                    if (g->nodes)
                        free(g->nodes);
                    free(g);
                    g = NULL;
                    return NULL;
                }
            }
        }
    }

    // deal with node
    for (i = 0; i < g->nnode; i++) {
        Onnx__NodeProto *node_proto = graph->node[i];
        if (!node_proto) break;
        g->nodes[i] = node_new(g, node_proto->name, op_map_onnx(node_proto->op_type));
        n = g->nodes[i];
        n->node_proto = node_proto;
        domain = node_proto->domain;
        if (!domain || (strlen(domain) == 0))
            domain = "ai.onnx";
        for (j = 0; j < ((Onnx__ModelProto *)(ctx->model))->n_opset_import; j++) {
            p = ((Onnx__ModelProto *)(ctx->model))->opset_import[j]->domain;
            if (!p || (strlen(p) == 0))
                p = "ai.onnx";
            if (strcmp(domain, p) == 0) {
                n->opset = ((Onnx__ModelProto *)(ctx->model))->opset_import[j]->version;
                break;
            }
        }
        if (node_proto->n_input > 0) {
            n->in = (tensor_t **)sys_malloc(node_proto->n_input * sizeof(tensor_t *));
            if (n->in) {
                n->nin = node_proto->n_input;
                for (j = 0; j < n->nin; j++) {
                    n->in[j] = context_get_tensor(ctx, node_proto->input[j]);
                }
            }
        }
        if (node_proto->n_output > 0) {
            n->out = (tensor_t **)sys_malloc(node_proto->n_output * sizeof(tensor_t *));
            if (n->out) {
                n->nout = node_proto->n_output;
                for (j = 0; j < n->nout; j++) {
                    n->out[j] = context_get_tensor(ctx, node_proto->output[j]);
                }
            }
        }
        if (!n->reshape)
            n->reshape = reshape_dummy;
        if (!n->operator)
            n->operator= operator_dummy;
    }

    return g;
}

context_t *load_onnx(struct serializer *s, const void *buf, int len) {
    context_t *ctx = NULL;
    if (!buf || len <= 0)
        return NULL;
    ctx = context_new(NULL);
    ctx->sez = s;
    ctx->model = onnx__model_proto__unpack(NULL, len, buf);
    ctx->model_size = len;
    ctx->name = sys_strdup("onnx");
    if (!ctx->model) {
        if (ctx)
            sys_free(ctx);
        return NULL;
    }
    ctx->tensor_map = hashmap_create();
    if (!ctx->tensor_map) {
        if (ctx->model)
            onnx__model_proto__free_unpacked(ctx->model, NULL);
        if (ctx)
            sys_free(ctx);
        return NULL;
    }
    // graph
    load_graph_onnx(ctx);
    return ctx;
}

static serializer_t onnx_serializer = {
    .fmt = "onnx",
    .load = load_onnx,
    .load_model = load_model_onnx,
    .load_tensor = load_tensor_onnx,
    .unload = unload_onnx,
    .load_graph = load_graph_onnx,
};

// ==================================================================================== //
//                                    serializer API
// ==================================================================================== //

serializer_t *serializer_new(const char *fmt) {
    if (strcmp(fmt, "onnx") == 0) {
        return &onnx_serializer;
    } else {  // default load by onnx
        LOG_WARN("Unsupport model format %s , use onnx as default\n", fmt);
        return &onnx_serializer;
    }
}

void serializer_free(serializer_t *sez) {
    if (sez) {
        sez->fmt = NULL;
        sez->load = NULL;
        sez->load_model = NULL;
        sez->load_graph = NULL;
        sez->unload = NULL;
        sez = NULL;
    }
}