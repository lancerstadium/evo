---
marp: true
_class: invert
paginate: true
math: mathjax
---
<!-- _header: 'Compute InkJet Lab' -->
<!-- _footer: evo | [Github](https://github.com/lancerstadium/evo/tree/ml) | [Docs](https://lancerstadium.github.io/evo/docs) -->

# 04 动态感知推理引擎

###### 作者：鲁天硕
###### 时间：2024/9/05

---

### Background 问题背景

目前，资源受限设备的推理部署存在一些挑战[1]：
1. 现有的大部分 DNN 研究集中关注**准确率**（*Accuracy*），在实际部署中常常需要权衡其他指标：每秒处理帧数、数据吞吐量、内存占用等等，如何在精度可接受的范围内提高其他指标是端侧部署主要目标；
2. 主流的静态模型压缩技术（量化、剪枝等）和 网络结构搜索技术（NAS），可以减少一定推理负载，但有**永久损害网络**预测能力的风险，它们消耗大量资源，且对于 MobileNet, SqueezeNet, ShuffleNet 这类参数冗余度小的网络的效果甚微；
3. 大部分推理数据常常存在**局部性与稀疏性**，现有的引擎框架无法针对不同的输入数据选择不同的处理方式，进一步收集**运行时信息**和**硬件信息**进行软硬件协同，通过**动态决策**改善推理性能。

---

### Data Sparsity 数据稀疏性

主流网络的中间层特征图[2]的稀疏性如下：

![alt text](image-2.png)

---

### Related Work 相关工作

按照决策级别可以分为如下四类：
1. Model-wise：...
2. Layer-wise：sact, SkipNet, ConvNet-AIG ...
3. Channel-wise：FalCon ...
4. Point-wise：...

> 优化决策：跳过（Skip），选择（Select），量化（Quant）等

![bg right w:240](image-1.png)
![bg right w:240](image.png)


---

### Dynamic-Aware 动态感知

感知数据来源：
1. 推理数据：特征图（Feature Map），参数量（Params）
2. 运行时剖析：实际时延（Actual Latency）
3. 硬件信息：时钟频率、内存占用
4. 环境数据：传感器采样的数据

> 注意：收集良好的数据是进行动态决策的前提

---

### Dynamic Hook Network 动态钩子网络

一种非侵入式的轻量网络结构，辅助主网络进行优化的动态决策，其生命周期如下：
1. 构建：在主网络文件导入时进行分析自动生成并挂载；
2. 训练：选择时机采样输入输出对，异步训练更新参数；
3. 推理：耦合进主网络同步推理，一般以门控形式决策；
4. 释放：主网络卸载后进行销毁或者保存（下次使用）。

---

### 参考文献

[1] Adapting Neural Networks at Runtime: Current Trends in At-Runtime Optimizations for Deep Learning
[2] [Dynamic runtime feature map pruning](https://arxiv.org/pdf/1812.09922)
