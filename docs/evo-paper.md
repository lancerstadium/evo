

## EVO Concept

AEP(*Autonomous Edge Learning and Inferencing Pipline*)

Memory Efficient Replacement Stratrgy

Binarized Neural Networks


### Background

近年来，物联网（Internet of Things）发展迅速，涵盖了智慧城市、智慧农业、工业化4.0、医疗保健、环境保护等领域。由于其部署领域广、设备数量庞大、产生数据量多的特性备受关注。

同时，机器学习是一种使用数据...
机器学习的基本流程：模型定义、模型训练、模型推理。

TinyML目的是赋予边缘侧设备计算能力，互联设备之间可以构成更大的系统，同时与云端交互。通过将计算任务下放到边缘侧... 

TinyML的挑战：
单款占用：
内存占用（Memory Footprint）
处理器功耗（Processing Power）

模型压缩四件套：量化（Quantization），减枝（Pruning），知识蒸馏（Knowledge Distillation），约束神经架构搜索（Constrained Neural Architecture Search）

TinyML框架：
- TFLM
- MicroTVM
- CMSIS-NN
- TinyEngine
- ...


### 引擎框架



### 支持二进制网络

BNN（Binary Nerual Network）

#### BNN算子设计

QBConv2D: 
QQConv2D:
BBConv2D:
BBPointwiseConv2D:
BMaxPool2D:
BBFC:
BBQFC:

### 软硬件协同设计

