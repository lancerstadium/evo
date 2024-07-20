

## EVO Concept

AEP(*Autonomous Edge Learning and Inferencing Pipline*)

Memory Efficient Replacement Stratrgy

Binarized Neural Networks


### 1 Background 研究背景

EVO: A Memory Effective TinyML Inference Engine

近年来，物联网（Internet of Things）发展迅速，涵盖了智慧城市、智慧农业、工业化4.0、医疗保健、环境保护等领域。由于其部署领域广、设备数量庞大、产生数据量多的特性备受关注。

同时，深度神经网络（DNN）已成为各个领域的关键方法，包括图像分类、语音识别和自然语言处理等等。推理是指在应用程序中执行经过训练的 AI 模型。它在各种实时应用中变得越来越普遍，例如交互式服务、自动驾驶汽车和智能可穿戴设备。实时推理任务通常以小批量甚至一批量的形式出现。这些任务对延迟敏感，需要低延迟来确保用户体验或安全性。因此，实时 AI 推理对吞吐量和时延都有很高的要求。

使用深度神经网络（DNN）的基本流程：模型定义、模型训练、模型推理。

TinyML目的是赋予边缘侧设备计算能力，互联设备之间可以构成更大的系统，同时与云端交互。通过将计算任务下放到边缘侧... 



### 2 TinyML 推理引擎概览


#### 2.1 推理引擎架构


#### 2.2 常见的推理引擎

TinyML框架：
- TFLM
- MicroTVM
- CMSIS-NN
- TinyEngine
- ...


#### 2.3 推理优化

推理引擎指标：
1. 推理速度（Inference Speed）
2. 吞吐量（Throughput）
3. 结果质量（Results Quality）
4. 内存占用（Mem Usage）
5. 可拓展（Extensibility）
6. 易用性（）


##### 2.3.1 data-level 数据级优化

1. 输入压缩（Input Compression）
2. 输出组织（Output Organization）

##### 2.3.2 model-level 模型级优化

高效结构设计（Efficient Structure Design）

模型压缩四件套：
1. 量化（Quantization）
2. 剪枝（Pruning）
3. 知识蒸馏（Knowledge Distillation）
4. 约束神经架构搜索（Constrained Neural Architecture Search）

##### 2.3.3 card-level (graph-level) 图优化

图拆分
跨核并行
数据布局
算子融合
批处理（提高内存重用）

##### 2.3.4 op-level 算子优化

内联汇编
硬件加速

##### 2.3.5 system-level 系统级优化

资源分配
加速器通信优化
提高CPU利用率和PCIe流量优化


#### 2.4 指标

GFLOPS


#### 2.5 挑战

TinyML的挑战：
- 内存占用（Memory Footprint）
- 处理器功耗（Processing Power）
- 内存带宽
- 大模型、复杂模型的推理


### 3 EVO 引擎框架概述


##### 3.1 Workloads 工作负载


#### 3.2 Architecture 引擎架构

EVO引擎架构：
- 模型压缩
- 模型优化
- 模型推理
- 硬件加速


### 4 Binary Nerual Network 二进制网络支持

BNN（Binary Neural Network）

#### 4.1 BNN 专用算子设计

QBConv2D: 
QQConv2D:
BBConv2D:
BBPointwiseConv2D:
BMaxPool2D:
BBFC:
BBQFC:



### 5 软硬件协同设计


拓展引擎对专用加速硬件的支持


### 6 性能分析与可视化



### Reference 参考文献

[First-Generation Inference Accelerator Deployment at Facebook](https://arxiv.org/pdf/2107.04140)
[A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294)