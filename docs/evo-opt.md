
## EVO Optimizer

### 1 Optimizer challenge

1. 结构冗余：算子融合、算子替换、常量折叠
2. 精度冗余：数据量化
3. 算法冗余：统一算子、计算图IR、提升kernel泛化
4. 读写冗余：数据排布优化、内存分配优化

优化流水线（*Pipline*）:

1. 获取计算图IR
2. 预优化：公共表达式消除、死代码消除、代数简化
3. 优化：算子融合、算子替换、常量折叠
4. 后优化：数据格式转换、内存布局计算、重复算子合并

### 2 Graph Optimizer

推理引擎（*Inference Engine*）: 主要采用基于模板的图优化，主要基于常用规则优化。

1. Basic: O1 常量折叠、O2 冗余节点消除、O3 算子融合
2. Extended: 特定后端优化（CPU/CUDA/NPU），针对硬件进行特殊复杂的Kernel融合。
3. Layout & Memory: 布局转换优化，数据重排优化。


#### 2.1 Basic Graph Opt

##### 2.1.1 O1: Constant Folding 常量折叠

- 离线预先确定输出值节点替换成常量
- ExpandDims 折叠
- Binary 折叠：


##### 2.1.2 O2: Redundant Node Elimination 冗余节点消除

- 算子冗余：算子替换（Op无意义）:
- Cast/Slice/...（Op参数无意义）:

##### 2.1.3 O3: Operator Fusion 算子融合

