
## Libevo

```
   $$$$$$\  $$\    $$\  $$$$$$\  
  $$  __$$\ \$$\  $$  |$$  __$$\ 
  $$$$$$$$ | \$$\$$  / $$ /  $$ |
  $$   ____|  \$$$  /  $$ |  $$ |
  \$$$$$$$\    \$  /   \$$$$$$  |
   \_______|    \_/     \______/ 

```

- `libevo` is a Inference engine used in TinyML write by pure C.
- **Keyword**: High Performance > Easy to Use > Compatibility

### 1 Docs


|    about    |              doc            |     desc     |
|:-----------:|:---------------------------:|:------------:|
|    model    | [evo-mdl.md](./evo-mdl.md)  |  model load  |
|  operator   | [evo-op.md](./evo-op.md)    | operator lib |
|    bindle   | [evo-bind.md](./evo-bind.md)|  Bindle API  |
|    tools    | [evo-tool.md](./evo-tool.md)|  Useful Tools |
|  reference  | [evo-ref.md](./evo-ref.md)  |  reference   |



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
    context_t * ctx = sez->load_model(sez, "model/mnist_8/model.onnx");
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