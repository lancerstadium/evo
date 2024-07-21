---
marp: true
_class: invert
paginate: true
---
<!-- _header: 'Compute InkJet Lab' -->
<!-- _footer: evo | [Github](https://github.com/lancerstadium/evo/tree/ml) | [Docs](https://lancerstadium.github.io/evo/docs) -->

# 02 领域专用的 TinyML 推理引擎

###### 作者：鲁天硕
###### 时间：2024/7/22

---

### 1.1 推理引擎分类

1. 通用场景：`TensorRT`, `OpenVINO`, `Mediapipe`, `OnnxRunTime`
2. 任务级专用：适用于一类任务的推理引擎
   1. 自然语言任务`NLP`：大语言 ...
   2. 图像处理任务`CV`：图像检测，图像识别 ...
3. 平台级专用：适用于一类平台的推理引擎
   1. 云端：（高性能）服务器，PC
   2. 边缘端：（低功耗）手机，MCU
4. 模型级专用：适用于一组模型的推理引擎
   1. 指定结构设计的模型：`CNN`, `RNN`, `Transformer`
   2. 资源消耗较低的模型：`MobileNet`, `ShuffleNet`

---

### 1.2 推理专用化

**案例**：四通道`CMYK`的图像处理模型进行边缘侧推理
1. 专用数据：按需求处理`CNYK`图片数据集
2. 专用算法：针对数据进行模型改进，增删改模块并训练
3. 专用数据级优化：
4. 专用模型级优化：对部署模型设计专用的文件格式，降低内存占用
5. 专用算子级优化：针对模型进行性能分析，提取专用算子
