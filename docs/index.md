---
title: "EVO"
author: LancerStadium
data: July 7, 2024
output: pdf_document
presentation:
    enableSpeakerNotes: true
    theme: serif.css
    width: 960
    height: 700
    slideNumber: true
    progress: false

---


## EVO

- `libevo` is a Inference engine used in TinyML write by pure C.
- **Keyword**: high perf, embedding, Deep Learning

### 1 Docs


|    about    |              doc            |     desc     |
|:-----------:|:---------------------------:|:------------:|
|    model    | [evo-mdl.md](./evo-mdl.md)  |  model load  |
|  operator   | [evo-op.md](./evo-op.md)    | operator lib |



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
    context_t * ctx = sez->load_file(sez, "model/mnist_8/model.onnx");
    tensor_t * t1 = context_get_tensor(ctx, "Input3");
    tensor_dump(t1);
    serializer_free(sez);
    // ...
    return 0;
}
```

### 4 Performance

- Task: Embedding Deep Learning Model
- Compare to: `TFLM`, `TVM` and so on
- BenchMark: `MLPerf`

### 5 Other Features

- Code scale:

```shell
make line   # calculate lines of code
```