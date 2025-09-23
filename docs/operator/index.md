## EVO Operator


### TODO Operator

|   Name   |            docs            |
|:--------:|:--------------------------:|
|  Pooling | [pool](./pool.md) |



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


|  Applied  |   Name   |  Detail  |
|:---------:|:--------:|:--------:|
| :material-check: | Abs | Support Type: `int8` |
|  | Acos | |
|  | Acosh | |
| :material-check: | Add | |
| :material-check: | And | |
| :material-check: | ArgMax | |
|  | ArgMin | |
|  | Asin | |
|  | Asinh | |
|  | Atan | |
|  | Atanh | |
| :material-check: | AveragePool | |
| :material-check: | BatchNormalization | |
|  | BitShift | |
| :material-check: | Cast | |
|  | Ceil | |
| :material-check: | Clip | |
|  | Compress | |
| :material-check: | Concat | |
|  | ConcatFromSequence | |
| :material-check: | Constant | |
| :material-check: | ConstantOfShape | |
| :material-check: | Conv | |
|  | ConvInteger | |
|  | ConvTranspose | |
| :material-check: | Cos | |
|  | Cosh | |
|  | CumSum | |
|  | DepthToSpace | |
|  | DequantizeLinear | |
|  | Det | |
| :material-check: | Div | |
|  | Dropout | |
|  | Einsum | |
|  | Elu | |
|  | Equal | |
|  | Erf | |
| :material-check: | Exp | |
| :material-check: | Expand | |
|  | EyeLike | |
| :material-check: | Flatten | |
|  | Floor | |
|  | GRU | |
| :material-check: | Gather | |
|  | GatherElements | |
|  | GatherND | |
| :material-check: | Gemm | |
|  | GlobalAveragePool | |
|  | GlobalLpPool | |
|  | GlobalMaxPool | |
|  | Greater | |
|  | HardSigmoid | |
|  | Hardmax | |
|  | Identity | |
|  | If | |
|  | InstanceNormalization | |
|  | IsInf | |
|  | IsNaN | |
|  | LRN | |
|  | LSTM | |
| :material-check: | LeakyRelu | |
|  | Less | |
| :material-check: | Log | |
|  | Loop | |
|  | LpNormalization | |
|  | LpPool | |
| :material-check: | MatMul | |
|  | MatMulInteger | |
|  | Max | |
|  | MaxPool | |
|  | MaxRoiPool | |
|  | MaxUnpool | |
| :material-check: | Mean | |
| :material-check: | Min | |
|  | Mod | |
| :material-check: | Mul | |
|  | Multinomial | |
| :material-check: | Neg | |
|  | NonMaxSuppression | |
|  | NonZero | |
|  | Not | |
|  | OneHot | |
|  | Or | |
| :material-check: | PRelu | |
|  | Pad | |
|  | Pow | |
|  | QLinearConv | |
|  | QLinearMatMul | |
|  | QuantizeLinear | |
|  | RNN | |
|  | RandomNormal | |
|  | RandomNormalLike | |
|  | RandomUniform | |
|  | RandomUniformLike | |
|  | Reciprocal | |
|  | ReduceL1 | |
|  | ReduceL2 | |
|  | ReduceLogSum | |
|  | ReduceLogSumExp | |
|  | ReduceMax | |
|  | ReduceMean | |
|  | ReduceMin | |
|  | ReduceProd | |
|  | ReduceSum | |
|  | ReduceSumSquare | |
|  | Relu | |
|  | Reshape | |
|  | Resize | |
|  | ReverseSequence | |
|  | RoiAlign | |
|  | Round | |
|  | Scan | |
|  | Scatter | |
|  | ScatterElements | |
|  | ScatterND | |
|  | Selu | |
|  | SequenceAt | |
|  | SequenceConstruct | |
|  | SequenceEmpty | |
|  | SequenceErase | |
|  | SequenceInsert | |
|  | SequenceLength | |
|  | Shape | |
|  | Shrink | |
|  | Sigmoid | |
|  | Sign | |
|  | Sin | |
|  | Sinh | |
|  | Size | |
|  | Slice | |
|  | Softplus | |
|  | Softsign | |
|  | SpaceToDepth | |
|  | Split | |
|  | SplitToSequence | |
|  | Sqrt | |
|  | Squeeze | |
|  | StringNormalizer | |
|  | Sub | |
|  | Sum | |
|  | Tan | |
| :material-check: | Tanh | |
|  | TfIdfVectorizer | |
|  | ThresholdedRelu | |
|  | Tile | |
|  | TopK | |
|  | Transpose | |
|  | Unique | |
|  | Unsqueeze | |
|  | Upsample | |
|  | Where | |
| :material-check: | Xor | |
|  | Celu | |
|  | DynamicQuantizeLinear | |
|  | GreaterOrEqual | |
|  | LessOrEqual | |
|  | LogSoftmax | |
|  | MeanVarianceNormalization | |
|  | NegativeLogLikelihoodLoss | |
|  | Range | |
|  | Softmax | |
|  | SoftmaxCrossEntropyLoss | |




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