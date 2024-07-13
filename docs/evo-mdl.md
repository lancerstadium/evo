<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

## EVO Model


### 1 Model Param

#### 1.1 Complex Analysis

1. 浮点运算次数（*FLOPs, Floating-point Operantion*）：计算量，衡量算法/模型时间复杂度。
2. 每秒执行浮点运算次数（*FLOPS, Floating-ponit Operations Per Second*）：计算速度，衡量硬件性能/速度的指标，即芯片算力。
3. 乘加操作次数（*MACCs, Multiply-accumulate Operations*）：MACCs大约是FLOPs的一半，将$w_0 * x_0$视为一次运算。
4. 模型参数（*Params*）：模型含有多少参数，直接决定模型大小和推理时内存占用（单位：MB），通常参数用float32表示，所以模型大小（单位：Byte）约为参数数量的4倍。
5. 内存访问代价（*MAC, Memory Access Cost*）：输入单个样本，模型/卷积完成一次前向传播发生的内存交换总量，即模型的空间复杂度（单位：Byte）。
6. 内存带宽：决定了数据从内存（vRAM）移动到计算核心的速度，是比计算速度更具代表性的指标。内存带宽值取决于内存和计算核心之间的数据传输速度，以及这两个部分之间总线中单独并行链路数量。

#### 1.2 Demo for Complex

用标准卷积层（*std conv*）贡献计算量：

- Params:
$$
    k_h * k_w * c_{in} * c_{out}
$$
- FLOPs:
$$
    k_h * k_w * c_{in} * c_{out} * H * W
$$


#### 1.2 Hardware Params

1. GPU：显存（vRAM）大小、显存带宽、计算核心数、计算速度（FLOPS）。
2. CPU：内存大小、内存带宽、计算核心数、计算速度（FLOPS）。


### 2 Model Format

#### 2.1 ONNX

1. ONNX(*Open Neural Network Exchange*): A binary file format which use `ProtoBuf` to serialize.

#### 2.2 tflite



### 3 Model Convert

#### 3.1 ProtoBuf


#### 3.2 FlatBuf