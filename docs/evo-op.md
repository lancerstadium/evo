## EVO Operator


### TODO Operator

|   Name   |           docs          |
|:--------:|:-----------------------:|
|  Pooling | [op-pool](./op-pool.md) |



### 1 Hardware Level Optimize


#### 1.1 Support CPU


| Name |   ISA   |  Company  |
|:----:|:-------:|:---------:|
|  AVX |  amd64  |   Intel   |
|  AMX |  amd64  |   Intel   |
| NEON | aarch64 |    Arm    |
|  RVV |  riscv  |    UCB    |



#### 1.2 Support GPU 


|  Name  |   ISA   |  Company  |
|:------:|:-------:|:---------:|
|  CUDA  |    |        |
| Vulkan |    |        |
| OpenCL |    |        |
|  Metal |    |        |



#### 1.3 Support NPU

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

total 162:

- [ ] Abs
- [ ] Acos
- [ ] Acosh
- [ ] Add
- [ ] And
- [ ] ArgMax
- [ ] ArgMin
- [ ] Asin
- [ ] Asinh
- [ ] Atan
- [ ] Atanh
- [ ] AveragePool
- [ ] BatchNormalization
- [ ] BitShift
- [ ] Cast
- [ ] Ceil
- [ ] Clip
- [ ] Compress
- [ ] Concat
- [ ] ConcatFromSequence
- [ ] Constant
- [ ] ConstantOfShape
- [ ] Conv
- [ ] ConvInteger
- [ ] ConvTranspose
- [ ] Cos
- [ ] Cosh
- [ ] CumSum
- [ ] DepthToSpace
- [ ] DequantizeLinear
- [ ] Det
- [ ] Div
- [ ] Dropout
- [ ] Einsum
- [ ] Elu
- [ ] Equal
- [ ] Erf
- [ ] Exp
- [ ] Expand
- [ ] EyeLike
- [ ] Flatten
- [ ] Floor
- [ ] GRU
- [ ] Gather
- [ ] GatherElements
- [ ] GatherND
- [ ] Gemm
- [ ] GlobalAveragePool
- [ ] GlobalLpPool
- [ ] GlobalMaxPool
- [ ] Greater
- [ ] HardSigmoid
- [ ] Hardmax
- [ ] Identity
- [ ] If
- [ ] InstanceNormalization
- [ ] IsInf
- [ ] IsNaN
- [ ] LRN
- [ ] LSTM
- [ ] LeakyRelu
- [ ] Less
- [ ] Log
- [ ] Loop
- [ ] LpNormalization
- [ ] LpPool
- [ ] MatMul
- [ ] MatMulInteger
- [ ] Max
- [ ] MaxPool
- [ ] MaxRoiPool
- [ ] MaxUnpool
- [ ] Mean
- [ ] Min
- [ ] Mod
- [ ] Mul
- [ ] Multinomial
- [ ] Neg
- [ ] NonMaxSuppression
- [ ] NonZero
- [ ] Not
- [ ] OneHot
- [ ] Or
- [ ] PRelu
- [ ] Pad
- [ ] Pow
- [ ] QLinearConv
- [ ] QLinearMatMul
- [ ] QuantizeLinear
- [ ] RNN
- [ ] RandomNormal
- [ ] RandomNormalLike
- [ ] RandomUniform
- [ ] RandomUniformLike
- [ ] Reciprocal
- [ ] ReduceL1
- [ ] ReduceL2
- [ ] ReduceLogSum
- [ ] ReduceLogSumExp
- [ ] ReduceMax
- [ ] ReduceMean
- [ ] ReduceMin
- [ ] ReduceProd
- [ ] ReduceSum
- [ ] ReduceSumSquare
- [ ] Relu
- [ ] Reshape
- [ ] Resize
- [ ] ReverseSequence
- [ ] RoiAlign
- [ ] Round
- [ ] Scan
- [ ] Scatter
- [ ] ScatterElements
- [ ] ScatterND
- [ ] Selu
- [ ] SequenceAt
- [ ] SequenceConstruct
- [ ] SequenceEmpty
- [ ] SequenceErase
- [ ] SequenceInsert
- [ ] SequenceLength
- [ ] Shape
- [ ] Shrink
- [ ] Sigmoid
- [ ] Sign
- [ ] Sin
- [ ] Sinh
- [ ] Size
- [ ] Slice
- [ ] Softplus
- [ ] Softsign
- [ ] SpaceToDepth
- [ ] Split
- [ ] SplitToSequence
- [ ] Sqrt
- [ ] Squeeze
- [ ] StringNormalizer
- [ ] Sub
- [ ] Sum
- [ ] Tan
- [ ] Tanh
- [ ] TfIdfVectorizer
- [ ] ThresholdedRelu
- [ ] Tile
- [ ] TopK
- [ ] Transpose
- [ ] Unique
- [ ] Unsqueeze
- [ ] Upsample
- [ ] Where
- [ ] Xor
- [ ] Celu
- [ ] DynamicQuantizeLinear
- [ ] GreaterOrEqual
- [ ] LessOrEqual
- [ ] LogSoftmax
- [ ] MeanVarianceNormalization
- [ ] NegativeLogLikelihoodLoss
- [ ] Range
- [ ] Softmax
- [ ] SoftmaxCrossEntropyLoss



### 4 learnable parameters

1. Conv:
    - kernel:   [1, 1, K_h, K_w]
    - bias  :   []
    - params:   (K_h * K_w * C_in + 0/1) * C_out
    - FLOPS :   (K_h * K_w * C_in + 0/1) * C_out * (H_out * W_out)
    - FLOPs :   2 * 

2. FC:
    - weight:   []
    - bias  :   []
    - params:   (C_in + 0/1) * C_out
    - FLOPS :   (C_in + 0/1) * C_out
    - FLOPs :   

3. BN:
    - scale:
    - shift:

4. Activation:
    - PRelu:


### 5 hyper parameters

1. learning rate
2. batch size
3. iterations
4. epochs

> data_size = 1200
> batch_size = 100
> epochs = 5
> update_count = (1200 / 100) * 5 = 60