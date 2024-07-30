

[Online Editer](https://www.overleaf.com/project/66a70c2de3101c3f34246913)


Title：Dynamic-Aware Inference Runtime on Resource-Constrained Edge Devices
题目：资源受限边缘设备上的动态感知推理运行时

## 摘要 Abstract

近年来，边缘人工智能（Edge AI）在许多领域取得了突破，但在边缘设备部署AI受到内存带宽、算力、能耗的限制，如何在资源受限设备上高效地推理是一个挑战。
为了利用推理数据空间稀疏性和局部性加速推理，本文提出了动态感知的推理运行时，可以免再训练的（retraining-free）加速模型推理。我们引入运行时剖析，通过准确率、时延、内存占用等实际指标进行决策，使用局部卷积、提前退出的方式进行推理优化。
我们在CIFAR、ImageNet上的实验表明，。

**关键词**：软硬件协同设计；边缘人工智能（edge-AI）；深度学习（DL）；模型压缩；神经加速器


## 1 介绍 Introduction

与基于云的计算相比，用于高效准确的边缘深度学习推理的硬件加速具有许多优势。通过减少在数据源附近处理数据时的延迟，可以实现快速、实时的用例。由于数据不传输到云端，因此确保了更好的安全性和带宽效率。边缘推理 (EI) 的其他好处是可扩展性和可靠性。因此，人工智能的许多应用正在从云端迁移到嵌入式边缘设备。 Alexa、Siri 和 Google Assistant 等虚拟助手使用片上语音识别作为实时协助处理的一部分。智能手机使用嵌入式人工智能，通过从多个镜头获取输入的计算成像技术来创建更好的图片。在智能电视中，人工智能会提升高清内容的质量，以重新创建缺失的细节。集成人工智能的可穿戴设备现在正在促进生命体征和健身信息的监测和处理，以跟踪或检测各种疾病。在医院环境中，边缘人工智能执行库存管理、患者远程监控、热筛查和疾病预测。无人机 (UAV) 可以通过人工智能促进的设备内处理确保远程和恶劣环境（交通、建筑、消防、制图、安全等）的安全检查。具有人工智能的机器人可在工业应用中提供具有高精度和可扩展性的高效制造。此外，可以使用人工智能相机检测制造缺陷以进行质量控制，而这在人眼中是不可能的。指纹检测、人脸识别安全、欺诈检测和自动驾驶是边缘人工智能带来的一些实际应用。

其结合了边缘计算和人工智能，使得设备能够在本地进行智能决策。深度神经网络（DNN）或深度学习（DL）的成功集成在许多领域取得了突破。然而，将这些高度准确的模型部署到最终用户应用程序的数据驱动、学习、自动和实用的机器学习 (ML) 解决方案仍然具有挑战性。深度学习算法通常计算成本高、耗电大，并且需要大量内存来处理数百万个参数的复杂迭代操作。因此，深度学习模型的训练和推理通常在云中的高性能计算 (HPC) 集群上执行。数据传输到云端会导致高延迟、往返延迟、安全和隐私问题以及无法实时决策。因此，在边缘设备上进行处理可以显着降低云端传输成本。边缘设备是最接近用户的终端设备，例如移动电话、网络物理系统 (CPS)、可穿戴设备、物联网 (IoT)、嵌入式和自治系统以及智能传感器。这些设备的内存、计算资源和功率处理能力有限。

深度学习越来越多地应用于广泛的产品和应用中，例如医学研究、预测性维护和工业环境中的质量控制。然而，大多数神经网络的大量计算和内存需求通常会阻碍其在大多数平台上的本地执行。大多数深度神经网络的研究都集中在提高准确性上，而不考虑模型的复杂性。随着社区转向更困难的问题——例如从分类到检测或姿态估计——架构的容量和计算复杂性往往会增加。然而，对于在手机、笔记本电脑或监控摄像头等消费设备上运行的实时应用程序来说，最重要的是性能（即每秒处理的帧数）和准确性之间的良好权衡。因此，许多行业依赖云后端进行昂贵的推理计算，这可能会带来延迟、稳定性、安全性和隐私限制。而静态优化技术，如剪枝、近似计算和量化，虽然可以实现推理工作负载的本地执行，但它们可能会永久损害神经网络的预测能力。于是，如何不修改原本的模型进行高效推理成为一项重要议题。在运行时内通过动态决策，还可以搭配静态优化技术使用，以达到最大化优化。

更实际的，在资源受限的边缘设备上部署时，通常需要对输入数据进行前处理和后处理，本运行时集成了图像加载和保存等功能，。

本文的主要贡献如下：


## 2 相关工作 Related work


同时，在大部分推理数据通常具有稀疏性和局部性。动态推理是一种新兴方法，它利用输入属性有选择地执行准确分类所需的显着计算子集。与永久删除神经元以提高模型效率的静态方法不同，动态方法仅根据输入实例暂时抑制计算。条件执行涉及网络模型的几个方面：
1. 组合网络规模缩放（Combined Network Size Scaling）：根据输入有条件地执行某些网络层或块。并非所有输入实例都需要所有分层计算才能正确分类[28]。在现代 DNN 中，重复的块构建在彼此之上以微调特征细节。较难的样本可能需要更深的嵌入才能准确分类，而较简单的样本可能只需要浅的嵌入。换句话说，较浅的推理对于更容易的样本是可行的，而对于更困难的情况则需要更深的层来保持性能如简单的图像需要比复杂的示例更深的网络。
2. 提前退出分支：后来的方法通过有条件地执行各个层来提高灵活性。这些方法基于残差架构对于层丢失具有鲁棒性的观察[20, 47]。 SkipNet [48] 使用强化学习来学习门控决策。 ConvNet-AIG [46] 使用 Gumbel-Softmax 技巧，而 BlockDrop [51] 使用强化学习训练单独的策略网络。
3. 动态稀疏化（Dynamic Sparsity）：动态稀疏性涉及训练网络以利用计算图的固有稀疏性。这是通过仅预测和识别应用 ReLU 激活函数产生的零元素来实现的，ReLU 激活函数常用于深度学习模型。通过这样做，动态稀疏性可以实现与剪枝类似甚至更高的计算节省，并且对预测精度的影响最小。这是因为网络和稀疏性诱导机制是联合训练的，并且重点是零元素，这不会影响网络的输出。总之，动态稀疏性提供了一种高效且有效的方法来降低神经网络的计算成本，而不牺牲预测性能

这些方法的一个共同属性是同一模型处理不同的输入实例。考虑到不同的实例具有独特的视觉特征，一个自然的问题就出现了：每个实例是否都需要所有级别的嵌入和同一组特征图才能准确分类？直观上，对于易于分类的图像来说，可能不需要更深的嵌入。因此，为了最大限度地提高计算效率，应仅为困难的输入实例保留与更深层相关的额外计算。此外，由于卷积通道/滤波器捕获特定于类的特征，因此可以通过在推理过程中跳过不相关的通道来节省不必要的计算。


