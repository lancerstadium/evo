
## Libevo

Welcome to Evo Page

{% assign date = '2020-04-13T10:20:00Z' %}

- Original date - {{ date }}
- With timeago filter - {{ date | timeago }}

- `libevo` is a Inference engine used in TinyML write by pure C.
- **Keyword**: Lite > High Performance > Easy to Use > Compatibility
- Try [online](./server)

![evo](./public/evo.svg)


### 1 Docs & PPT

#### 1.1 Docs

|    about    |              doc                |     desc     |
|:-----------:|:-------------------------------:|:------------:|
|    model    | [evo-mdl.md](./model.md)        |  model load  |
|  operator   | [evo-op.md](./operator/index.md)| operator lib |
|    bindle   | [evo-bind.md](./bind.md)        |  Bindle API  |
|    tools    | [evo-tool.md](./tool.md)        |  Useful Tools|
|   profile   | [evo-profile.md](./profile.md)  |   profiler   |
|    quant    | [evo-quant.md](./quant.md)      |   quant      |
|  reference  | [evo-ref.md](./reference.md)    |  reference   |


#### 1.1 PPT

|     about     |              ppt            |     desc     |
|:-------------:|:---------------------------:|:------------:|
|  Infer Engine |  [ppt-01](./ppt-01.html)    |    Engine    |
|  Domin Engine |  [ppt-02](./ppt-02.html)    |    Engine    |
|  Dynamic Eng  |  [ppt-04](./ppt-04.html)    |    Engine    |
|  Model Quant  |  [ppt-05](./ppt-05.html)    |    Model     |
|  Infer Intro  |  [ppt-06](./ppt-06.html)    |    Engine    |
|  Runtime Opt  |  [ppt-07](./ppt-07.html)    |    Runtime   |
|  Memory Org   |  [ppt-08](./ppt-08.html)    |    Memory    |

### 2 Build

- To build the item, you should:

```shell
make
```


### 3 Demo

- Here is a c demo for `libevo` :

```c
#include <evo.h>

int main() {
    // ...
    serializer_t * sez = serializer_get("onnx");
    model_t * mdl = sez->load_file(sez, "model/mnist_8/model.onnx");
    tensor_t * t1 = model_get_tensor(mdl, "Input3");
    tensor_dump(t1);
    // ...
    return 0;
}
```

### 4 Performance

- Task: Embedding Deep Learning Model
- Compare to: `ORT`, `TFLM`, `TVM` and so on
- BenchMark: `MLPerf`



### 5 Other Features

- Code scale:

```shell
make line   # calculate lines of code
```