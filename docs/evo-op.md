## EVO Operator


### TODO Operator

|   Name   |           docs          |
|:--------:|:-----------------------:|
|  Pooling | [op-pool](./op-pool.md) |



### 1 Hardware Level Optimize


#### 1.1 CPU Operator


| Name |   ISA   |  Company  |
|:----:|:-------:|:---------:|
|  AVX |  amd64  |   Intel   |
|  AMX |  amd64  |   Intel   |
| NEON | aarch64 |    Arm    |
|  RVV |  riscv  |    UCB    |



#### 1.2 GPU 


|  Name  |   ISA   |  Company  |
|:------:|:-------:|:---------:|
|  CUDA  |    |        |
| Vulkan |    |        |
| OpenCL |    |        |
|  Metal |    |        |



#### 1.3 NPU

|  Name  |   ISA   |  Company  |
|:------:|:-------:|:---------:|
| CoreML |    |        |
|  HIAI  |    |        |
|  NNAPI |    |        |





### 2 Hign Perfermance Operator Lib






|  Name  |   ISA   |  Company  |
|:------:|:-------:|:---------:|
| cuDNN  |    |        |
| MKLDNN |    |        |


### 3 onnx operator

```txt
    case 0x0b87d47b: /* "Abs" */
    case 0x7c82680b: /* "Acos" */
    case 0x0ccf69d3: /* "Acosh" */
    case 0x0b87d4ae: /* "Add" */
    case 0x0b87d5f8: /* "And" */
    case 0xa7c70ea5: /* "ArgMax" */
    case 0xa7c70fa3: /* "ArgMin" */
    case 0x7c82ab50: /* "Asin" */
    case 0x0cd815b8: /* "Asinh" */
    case 0x7c82ae89: /* "Atan" */
    case 0x0cd88011: /* "Atanh" */
    case 0xf1a1e23a: /* "AveragePool" */
    case 0x2d3b46ee: /* "BatchNormalization" */
    case 0x0bfe45a2: /* "BitShift" */
    case 0x7c8378d0: /* "Cast" */
    case 0x7c838882: /* "Ceil" */
    case 0x7c83a64d: /* "Clip" */
    case 0xb7db9db1: /* "Compress" */
    case 0xac3f4a9d: /* "Concat" */
    case 0x5053caca: /* "ConcatFromSequence" */
    case 0xba6816ef: /* "Constant" */
    case 0xe468a875: /* "ConstantOfShape" */
    case 0x7c83b3bb: /* "Conv" */
    case 0x8371dbe9: /* "ConvInteger" */
    case 0x3903c4ba: /* "ConvTranspose" */
    case 0x0b87deaa: /* "Cos" */
    case 0x7c83b452: /* "Cosh" */
    case 0xacab0fbf: /* "CumSum" */
    case 0xc9c1d669: /* "DepthToSpace" */
    case 0xf9cc985a: /* "DequantizeLinear" */
    case 0x0b87e1a2: /* "Det" */
    case 0x0b87e228: /* "Div" */
    case 0x883bca72: /* "Dropout" */
    case 0xb07d4f76: /* "Einsum" */
    case 0x0b87e6cb: /* "Elu" */
    case 0x0d1f905d: /* "Equal" */
    case 0x0b87e782: /* "Erf" */
    case 0x0b87e852: /* "Exp" */
    case 0xb18d8a45: /* "Expand" */
    case 0xe4c1560d: /* "EyeLike" */
    case 0x13363dd3: /* "Flatten" */
    case 0x0d2ed347: /* "Floor" */
    case 0x0b87ebd3: /* "GRU" */
    case 0xb499f620: /* "Gather" */
    case 0x7c94d43d: /* "GatherElements" */
    case 0x42f00872: /* "GatherND" */
    case 0x7c85ba8b: /* "Gemm" */
    case 0x9289c84b: /* "GlobalAveragePool" */
    case 0x3f5a29ac: /* "GlobalLpPool" */
    case 0x575f0fb6: /* "GlobalMaxPool" */
    case 0x6e6d652f: /* "Greater" */
    case 0x10341df0: /* "HardSigmoid" */
    case 0x94acb4aa: /* "Hardmax" */
    case 0xdfd9b28f: /* "Identity" */
    case 0x00597414: /* "If" */
    case 0xfb0902c1: /* "InstanceNormalization" */
    case 0x0d68519e: /* "IsInf" */
    case 0x0d68651e: /* "IsNaN" */
    case 0x0b880111: /* "LRN" */
    case 0x7c882885: /* "LSTM" */
    case 0xea2c5c33: /* "LeakyRelu" */
    case 0x7c88793c: /* "Less" */
    case 0x0b8804e7: /* "Log" */
    case 0x7c88a33f: /* "Loop" */
    case 0x07f77ce8: /* "LpNormalization" */
    case 0xc13f923b: /* "LpPool" */
    case 0xc2987915: /* "MatMul" */
    case 0x62fbd803: /* "MatMulInteger" */
    case 0x0b88076b: /* "Max" */
    case 0x15f18a25: /* "MaxPool" */
    case 0x018c06cf: /* "MaxRoiPool" */
    case 0x641501e8: /* "MaxUnpool" */
    case 0x7c890346: /* "Mean" */
    case 0x0b880869: /* "Min" */
    case 0x0b880925: /* "Mod" */
    case 0x0b8809f3: /* "Mul" */
    case 0xaec55410: /* "Multinomial" */
    case 0x0b880c1f: /* "Neg" */
    case 0x254e25a1: /* "NonMaxSuppression" */
    case 0x82e45c50: /* "NonZero" */
    case 0x0b880d76: /* "Not" */
    case 0xc825b932: /* "OneHot" */
    case 0x005974e6: /* "Or" */
    case 0x0dd55b8d: /* "PRelu" */
    case 0x0b88141a: /* "Pad" */
    case 0x0b8815fb: /* "Pow" */
    case 0xe569f427: /* "QLinearConv" */
    case 0xfe108481: /* "QLinearMatMul" */
    case 0x37138211: /* "QuantizeLinear" */
    case 0x0b881a13: /* "RNN" */
    case 0xc100684f: /* "RandomNormal" */
    case 0xa0b57174: /* "RandomNormalLike" */
    case 0xf8e97c66: /* "RandomUniform" */
    case 0x10a8b90b: /* "RandomUniformLike" */
    case 0x73d06f69: /* "Reciprocal" */
    case 0x7944853a: /* "ReduceL1" */
    case 0x7944853b: /* "ReduceL2" */
    case 0xeab46d14: /* "ReduceLogSum" */
    case 0x9a057a01: /* "ReduceLogSumExp" */
    case 0xa1d53763: /* "ReduceMax" */
    case 0xdc7c323e: /* "ReduceMean" */
    case 0xa1d53861: /* "ReduceMin" */
    case 0xdc7e1072: /* "ReduceProd" */
    case 0xa1d55372: /* "ReduceSum" */
    case 0x20917223: /* "ReduceSumSquare" */
    case 0x7c8bc29d: /* "Relu" */
    case 0x9fdbcf8d: /* "Reshape" */
    case 0xce8a9197: /* "Resize" */
    case 0x5d77301a: /* "ReverseSequence" */
    case 0x830cb9da: /* "RoiAlign" */
    case 0x0e09b7cd: /* "Round" */
    case 0x7c8c450a: /* "Scan" */
    case 0xe6ece5fb: /* "Scatter" */
    case 0xb4db6f18: /* "ScatterElements" */
    case 0x55be5b0d: /* "ScatterND" */
    case 0x7c8c4efe: /* "Selu" */
    case 0xe537ccd3: /* "SequenceAt" */
    case 0xa52772e3: /* "SequenceConstruct" */
    case 0x5e6e772d: /* "SequenceEmpty" */
    case 0x5e70f50e: /* "SequenceErase" */
    case 0x35a57cb3: /* "SequenceInsert" */
    case 0x3bff64e0: /* "SequenceLength" */
    case 0x0e17a4d6: /* "Shape" */
    case 0xd11575d4: /* "Shrink" */
    case 0xf5548151: /* "Sigmoid" */
    case 0x7c8c5f56: /* "Sign" */
    case 0x0b8821ef: /* "Sin" */
    case 0x7c8c6037: /* "Sinh" */
    case 0x7c8c61c0: /* "Size" */
    case 0x0e19f6b5: /* "Slice" */
    case 0x6bec36a5: /* "Softplus" */
    case 0x6bedcd32: /* "Softsign" */
    case 0xa4436289: /* "SpaceToDepth" */
    case 0x0e1c35d1: /* "Split" */
    case 0x50e66fcd: /* "SplitToSequence" */
    case 0x7c8c82cf: /* "Sqrt" */
    case 0x08f69207: /* "Squeeze" */
    case 0xf404645f: /* "StringNormalizer" */
    case 0x0b88236f: /* "Sub" */
    case 0x0b88237a: /* "Sum" */
    case 0x0b882528: /* "Tan" */
    case 0x7c8cca90: /* "Tanh" */
    case 0x46fbf3df: /* "TfIdfVectorizer" */
    case 0xa646ea33: /* "ThresholdedRelu" */
    case 0x7c8cec53: /* "Tile" */
    case 0x7c8d0643: /* "TopK" */
    case 0x940b3944: /* "Transpose" */
    case 0xd6278d9c: /* "Unique" */
    case 0xc836156a: /* "Unsqueeze" */
    case 0xae63c66c: /* "Upsample" */
    case 0x0e601820: /* "Where" */
    case 0x0b8837fe: /* "Xor" */
    case 0x7c8388ee: /* "Celu" */
    case 0x718dbc56: /* "DynamicQuantizeLinear" */
    case 0x7b2541c8: /* "GreaterOrEqual" */
    case 0x60d9a535: /* "LessOrEqual" */
    case 0xf8c82769: /* "LogSoftmax" */
    case 0xbb8f2396: /* "MeanVarianceNormalization" */
    case 0x6ed111df: /* "NegativeLogLikelihoodLoss" */
    case 0x0e01ebd2: /* "Range" */
    case 0x034529c7: /* "Softmax" */
    case 0x522154a3: /* "SoftmaxCrossEntropyLoss" */
```