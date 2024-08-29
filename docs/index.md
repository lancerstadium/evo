---
---


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

|    about    |              doc                    |     desc     |
|:-----------:|:-----------------------------------:|:------------:|
|    model    | [evo-mdl.md](./evo-mdl.md)          |  model load  |
|  operator   | [evo-op.md](./evo-op.md)            | operator lib |
|    bindle   | [evo-bind.md](./evo-bind.md)        |  Bindle API  |
|    tools    | [evo-tool.md](./evo-tool.md)        |  Useful Tools|
|   profile   | [evo-profile.md](./evo-profile.md)  |   profiler   |
|  reference  | [evo-ref.md](./evo-ref.md)          |  reference   |


#### 1.1 PPT

|     about     |              ppt            |     desc     |
|:-------------:|:---------------------------:|:------------:|
|  Infer Engine |  [ppt-01](./ppt-01.html)    |    Engine    |
|  Domin Engine |  [ppt-02](./ppt-02.html)    |    Engine    |


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
    serializer_t * sez = serializer_new("onnx");
    model_t * mdl = sez->load_model(sez, "model/mnist_8/model.onnx");
    tensor_t * t1 = model_get_tensor(mdl, "Input3");
    tensor_dump(t1);
    serializer_free(sez);
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