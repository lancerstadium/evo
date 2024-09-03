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

<div align="center">
<h2> 
<i>Evo</i>: Dynamic-Aware Edge Inference Framework on Resource-Constrained Devices <br> 资源受限设备上的动态感知边缘推理框架 
</h2>
</div>

<div align="center"> 
<a href="">Lu TianShuo</a><sup>1</sup>, XXX<sup>1*</sup>, XXX<sup>1*†</sup>
</div>

<p align="center"> <sup>1</sup>JiangNan University, <sup>2</sup>XXX </p>

<p align="center">
<a href="https://arxiv.org/abs/XXXX.XXXXX" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/lancerstadium/evo/blob/ml/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-MIT-%23B7A800" /></a>
<a href="https://github.com/lancerstadium/evo/tree/ml">
    <img src="https://img.shields.io/badge/Code-Github-blue" /></a>
<a href="https://lancerstadium.github.io/evo/docs/">
    <img src="https://img.shields.io/badge/Docs-Online-8A2BE2" /></a>
<a href="https://lancerstadium.github.io/evo/docs/evo-paper-little">
    <img src="https://img.shields.io/badge/Paper-Online-FF8C00" /></a>
</p>

<b>Overview:</b> 
<div style="text-align: justify;">
Evo is a Dynamic-Aware engine which use for Edge Inference...
</div>

<div align="center"> 
<img src="./public/logo.svg" width=25% height=25% class="center" alt="Opt-Arch">
</div>

---

## 0 摘要 Abstract

近年来，边缘人工智能（Edge AI）在许多领域取得了突破，但在边缘设备部署AI受到内存带宽、算力、能耗的限制，如何在资源受限设备上高效地推理是一个挑战。
为了利用推理数据空间稀疏性和局部性加速推理，本文提出了动态感知的推理引擎，可以免再训练的（retraining-free）加速模型推理。我们引入运行时剖析（Runtime-Profile），通过准确率、时延、内存占用等实际指标进行决策，使用局部卷积、提前退出的方式进行推理优化。
我们使用CIFAR-10、ImageNet等数据集在推理引擎内进行了识别、目标检测、图像超分实验，结果表明：。

**关键词**：软硬件协同设计（Co-Design）；边缘人工智能（Edge-AI）；深度学习（Deep Learning）；模型压缩；神经加速器


---
## 1 介绍 Introduction


最近，AIot与TinyML的兴起使得人工智能部署逐步从云端迁移到嵌入式边缘设备。依靠低功耗的深度学习专用推理硬件，边缘推理（Edge Inference, EI）减少在数据源附近处理数据时的延迟，可以实现快速、实时的深度学习推理部署。相比于昂贵的云端推理，边缘推理确保了更好的稳定性、安全性和带宽效率。同时，边缘推理具有良好的可扩展性。

边缘推理在各个领域已有广泛的应用。CV任务领域：小目标检测、小目标分割、图像超分、图像纹理增强、暗光增强。Alexa、Siri 和 Google Assistant 等虚拟助手使用片上语音识别作为实时协助处理的一部分。智能手机使用嵌入式人工智能，通过从多个镜头获取输入的计算成像技术来创建更好的图片。在智能电视中，人工智能会提升高清内容的质量，以重新创建缺失的细节。集成人工智能的可穿戴设备现在正在促进生命体征和健身信息的监测和处理，以跟踪或检测各种疾病。在医院环境中，边缘人工智能执行库存管理、患者远程监控、热筛查和疾病预测。无人机 (UAV) 可以通过人工智能促进的设备内处理确保远程和恶劣环境（交通、建筑、消防、制图、安全等）的安全检查。具有人工智能的机器人可在工业应用中提供具有高精度和可扩展性的高效制造。此外，使用摄像头搭配深度学习网络来进行检测制造缺陷以进行质量控制、指纹检测、人脸识别安全、欺诈检测和自动驾驶是边缘人工智能带来的一些实际应用。

其结合了边缘计算和人工智能，使得设备能够在本地进行智能决策。深度神经网络（DNN）或深度学习（DL）的成功集成在许多领域取得了突破。然而，将这些高度准确的模型部署到最终用户应用程序的数据驱动、学习、自动和实用的机器学习 (ML) 解决方案仍然具有挑战性。深度学习算法通常计算成本高、耗电大，并且需要大量内存来处理数百万个参数的复杂迭代操作。因此，深度学习模型的训练和推理通常在云中的高性能计算 (HPC) 集群上执行。数据传输到云端会导致高延迟、往返延迟、安全和隐私问题以及无法实时决策。因此，在边缘设备上进行处理可以显着降低云端传输成本。边缘设备是最接近用户的终端设备，例如移动电话、网络物理系统 (CPS)、可穿戴设备、物联网 (IoT)、嵌入式和自治系统以及智能传感器。这些设备的内存、计算资源和功率处理能力有限。

深度学习越来越多地应用于广泛的产品和应用中，例如医学研究、预测性维护和工业环境中的质量控制。然而，大多数神经网络的大量计算和内存需求通常会阻碍其在大多数平台上的本地执行。

在资源受限的设备上推理部署时，也存在一些挑战：
1. 大多数深度神经网络的研究都集中在提高准确性上，而不考虑模型的复杂性。随着社区转向更困难的问题——例如从分类到检测或姿态估计——架构的容量和计算复杂性往往会增加。然而，对于在手机、笔记本电脑或监控摄像头等消费设备上运行的实时应用程序来说，最重要的是性能（即每秒处理的帧数）和准确性之间的良好权衡。
2. 主流的静态优化技术，如剪枝、近似计算和量化，虽然可以实现推理工作负载的本地执行，但它们可能会永久损害神经网络的预测能力。于是，如何不修改原本的模型进行高效推理成为一项重要议题。在运行时内通过动态决策，还可以搭配静态优化技术使用，以达到最大化优化。
3. 更实际的，部署程序通常需要对输入数据（图像、语音、文本等数据）进行前处理和后处理，这些数据处理库通常比较功能复杂且占用资源较多，不适用于嵌入式部署。

本文的主要贡献如下：



---
## 2 相关工作 Related work

传统的静态推理优化：一般输入模型文件，输出优化后的模型文件，剪枝、量化等等。

同时，在大部分推理数据通常具有稀疏性和局部性。动态推理是一种新兴方法，它利用输入属性有选择地执行准确分类所需的显着计算子集。与永久删除神经元以提高模型效率的静态方法不同，动态方法仅根据输入实例暂时抑制计算。条件执行涉及网络模型的几个方面：
1. 组合网络规模缩放（Combined Network Size Scaling）：根据输入有条件地执行某些网络层或块。并非所有输入实例都需要所有分层计算才能正确分类[28]。在现代 DNN 中，重复的块构建在彼此之上以微调特征细节。较难的样本可能需要更深的嵌入才能准确分类，而较简单的样本可能只需要浅的嵌入。换句话说，较浅的推理对于更容易的样本是可行的，而对于更困难的情况则需要更深的层来保持性能如简单的图像需要比复杂的示例更深的网络。
2. 提前退出分支（Early Exit Branch）：后来的方法通过有条件地执行各个层来提高灵活性。这些方法基于残差架构对于层丢失具有鲁棒性的观察[20, 47]。 SkipNet [48] 使用强化学习来学习门控决策。 ConvNet-AIG [46] 使用 Gumbel-Softmax 技巧，而 BlockDrop [51] 使用强化学习训练单独的策略网络。
3. 测试时修剪（Pruning at test time）测试时的剪枝侧重于在推理阶段从神经网络中删除不必要的操作，而不在训练阶段调整网络
4. 动态稀疏化（Dynamic Sparsity）：动态稀疏性涉及训练网络以利用计算图的固有稀疏性。这是通过仅预测和识别应用 ReLU 激活函数产生的零元素来实现的，ReLU 激活函数常用于深度学习模型。通过这样做，动态稀疏性可以实现与剪枝类似甚至更高的计算节省，并且对预测精度的影响最小。这是因为网络和稀疏性诱导机制是联合训练的，并且重点是零元素，这不会影响网络的输出。总之，动态稀疏性提供了一种高效且有效的方法来降低神经网络的计算成本，而不牺牲预测性能

上述方法都需要对模型进行修改并重新训练。这些方法的一个共同属性是同一模型处理不同的输入实例。考虑到不同的实例具有独特的视觉特征，一个自然的问题就出现了：每个实例是否都需要所有级别的嵌入和同一组特征图才能准确分类？直观上，对于易于分类的图像来说，可能不需要更深的嵌入。因此，为了最大限度地提高计算效率，应仅为困难的输入实例保留与更深层相关的额外计算。此外，由于卷积通道/滤波器捕获特定于类的特征，因此可以通过在推理过程中跳过不相关的通道来节省不必要的计算。


### 2.1 前向训练
Hinton 于 2022 年提出了前向前向 (Forward Forward, FF) 算法，该算法提供了一种有效的分层学习方法，用两次前向传递取代了传统的反向传播。

相比于传统反向传播（Backward, BP）的优势：
1. 贴近生物表达：与生物神经系统类似，FF 的学习过程基于直接调整神经元活动（增强或减少活动）以响应不同类型的输入模式。
2. 支持局部学习：
3. 对硬件友好：从硬件实现的角度来看，FF 消除了在每个模块计算后存储中间激活的必要性，这显着降低了训练期间的内存需求。这有利于许多深度网络架构中的模型并行性，以实现更快的训练和推理。

前向前向算法的补充。 FF的核心思想是用两次前向传递代替反向传播的前向和后向传递，并针对具有相反目标的两类数据（即正数据和负数据）操纵优度函数的优化。对于构建的正数据，FF 训练鼓励调整权重以增加每个隐藏层的优点。相反，对于负数据，它调整权重以降低优度函数。

对于监督图像分类任务，FF 操纵输入图像来生成正样本和负样本。对于每个输入图像 $x\in\mathbb{R}^{m\times1}$，它将 $x$ 的前 $K$ 个像素替换为正确（Positive）或不正确（Negtive）的 独热码（*One-hot*） 标签 $y\in\mathbb{R}^{k\times1}$。这个过程创建了表示为 $x^{*}$ 的修改模式，其中 $*\in\{pos,neg\}$ 指示标签向量是正还是负。优度函数（Goodness Function） $g$ 可以形式化如下：

$$
v^*=Wx^*,\quad g^*=\|v^*\|_2^2,\quad Loss=\sigma(g^{pos}-\theta)+\sigma(\theta-g^{neg})
$$

FF Loss 简写：

$$
L(x)=\log(1+e^{y(\theta-G_x)})
$$


其中， $W\in\mathbb{R}^{n\times m}$ 表示连接前一层到当前层的矩阵，$v$ 表示加权输入和，$\theta$ 是调节优度敏感度的阈值参数。给定优度函数（Goodness）的表达式， FF 采用负对数sigmoid函数 $\sigma(x)=\log(1+\exp(-x))$，通过同时反向调整 $\sigma(g^{pos}-\theta)$ 和 $\sigma(\theta-g^{neg})$ 来优化网络参数方向。

SymBa Loss 简写：

$$
L(P,N)=\log(1+e^{-\alpha(G_P-G_N)})
$$



---
## 3 方法 Method

### 3.1 架构概览

相关概念：
- 数据稀疏性（*Data Sparsity*）：在模型前向计算时，经过特定层会产生大量含有0数据的张量，对于这些稀疏张量，可分析其计算特征进行深度优化。
- 运行时剖析（*Runtime Profile*）：在推理引擎内部对模型参数、运行时信息、设备信息进行采样，用于后续推理优化。
- 动态钩子网络（*Dynamic Hook Network*）：动态钩子网络与主网络协同计算，负责对主网络数据进行采样、分析以及执行优化决策。
- 空间执行掩码（*Spatial Execution Masks*）：用于裁剪窗口，进行后续计算优化
- 动态感知推理（*Dynamic-Aware Inference*）：数据感知 --> 构筑动态钩子网络 --> 执行决策


### 3.2 动态钩子网络

设计原则：非侵入式推理网络，在主网络进行前向推理过程时挂载，目的是用于优化主网络相关指标进行动态优化

特征（features）：
1. 自适应构建：通过主网络信息进行分析动态构建钩子网络，一般为轻量化网络
2. 同步推理：在模型推理时，耦合进主网络协同推理，一般推理消耗时间较少
3. 异步训练：采样主网络输入输出对，在主网络执行完毕后通过前向训练的方式对网络参数进行局部学习


### 3.3 感知量化 & 重参数化

- [Awesome quantization](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) 开头写的不错

- [Instance-Aware Dynamic Neural Network Quantization](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Instance-Aware_Dynamic_Neural_Network_Quantization_CVPR_2022_paper.pdf) | [CSDN](https://blog.csdn.net/Z960515/article/details/139701036)

- [Improved Model Design and Training Techniques for Efficient DNN Inference](https://repositories.lib.utexas.edu/server/api/core/bitstreams/076f3f54-4fc7-4219-b6c9-6f7f3547b3d9/content)

- [Soft Threshold Weight Reparameterization for Learnable Sparsity](http://proceedings.mlr.press/v119/kusupati20a/kusupati20a.pdf)

- [DAQ: Channel-Wise Distribution-Aware Quantization for Deep Image Super-Resolution Networks](https://openaccess.thecvf.com/content/WACV2022/papers/Hong_DAQ_Channel-Wise_Distribution-Aware_Quantization_for_Deep_Image_Super-Resolution_Networks_WACV_2022_paper.pdf) 图画的不错

- [CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670360.pdf) 数据相关


### 3.4 层次跳过

自适应层级动态钩子网络（Adaptive Layer Dynamic Hook Network, AL-DHN）组件设计：对于残差网络进行跳过处理

- 输入：前一层的输出 $C×H×W$
- 头（Head）：第一个组件负责将输入特征图规范化为统一大小，一般采用 GlobalAreragePool 算子将输入特征压缩为 $C×1×1$ 通道描述符
- 体（Body）：第二个组件有效地估计当前图像的各个层的相关性。为捕获通道之间的依赖关系，ConvNet-AIG 添加了一个由两个全连接层（fc）组成的简单非线性函数，这些层通过 BatchNorm 和 ReLU 连接，计算相关性分数 $\beta=\mathbf{W}_2\sigma(\mathbf{W}_1\mathbf{z})$ ，其中 $\mathbf{W}_1\in\mathbb{R}^{d\times C},\:\mathbf{W}_2\in\mathbb{R}^{2\times d}$ 。（对于向量 $\beta$ ，分别包含用于计算和跳过下一层动作的两个非归一化分数。一般来说，如果执行得分 $\beta_1$ 大于跳过该层的得分（即 $\beta_0$），则认为该层与给定输入相关。）
- 门（Gate）：第三个组件通过特殊设计的网络输出决策信息。一般使用 GumbelSoftmax 进行采样来做出离散决策。


### 3.5 通道剪枝

自适应通道动态钩子网络（Adaptive Channel Dynamic Hook Network, AC-DHN）设计：

1. 采样（Sampling）：Middle Feature Map -> Gumble Softmax -> Mask
2. 排序（Sorting）：Mask -> Argsort -> Top K Index of Sparse Attention (Other for Conv)


### 3.6 稀疏卷积

自适应元素动态钩子网络（Adaptive Element Dynamic Hook Network, AL-DHN）组件设计：与参数稀疏性相比，特征图稀疏性与每个输入相关，具有更好的适应性。实际的稀疏模式是非结构性的，并且随机位于具有不同形状的特征图上。

### 3.7 提前退出






---
## 4 实验 Experiment


### 4.1 实验环境

数据集：
1. CIFAR-10
2. ImageNet


### 4.2 实验结果

指标：
1. 吞吐量（）
2. 内存占用（）
3. 每秒浮点计算次数（GFLOPS）
4. 推理时间（FPS）

---
## 5 结论 Conclusion


后续优化：
对于决策的信息收集可拓展到特定硬件平台

---
## 参考文献 References

> 动态推理优化相关文献：
> 1. 2019 DYNAMIC RUNTIME FEATURE MAP PRUNING
> 2. Adapting Neural Networks at Runtime: Current Trends in At-Runtime Optimizations for Deep Learning
> 3. FalCon: Fine-grained Feature Map Sparsity Computing with Decomposed Convolutions for Inference Optimization
> 4. Fully Dynamic Inference with Deep Neural Networks
> 5. Spatially Adaptive Computation Time for Residual Networks
> 6. Convolutional Networks with Adaptive Inference Graphs：残差网络的空间自适应计算时间 ResNet 自适应地确定在哪一层之后停止计算。
> 7. A Heterogeneous Dynamic Convolutional Neural Network for Image Super-resolution

> 前向训练参考文献：
> 1. The Forward-Forward Algorithm: Some Preliminary Investigations
> 2. SymBa: Symmetric Backpropagation-Free Contrastive Learning with Forward-Forward Algorithm for Optimizing Convergence
> 3. Distance-Forward Learning: Enhancing the Forward-Forward Algorithm Towards High-Performance On-Chip Learning

---
## 附录

[Online Editer](https://www.overleaf.com/project/66a70c2de3101c3f34246913)