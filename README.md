# evo

<img src="./docs/public/logo.svg" width="320" align="center">

- Docs: [evo-docs](https://lancerstadium.github.io/evo/docs/)
- Server: [evo-server](https://lancerstadium.github.io/evo/docs/server)

## 1 Build & Test

### 1.1 Build

```shell
# build
make
# clean
make clean
```

### 1.2 Tests

```shell
# test all
make test
# test target xxx file
make test TEST_TRG=xxx_tests
# clean build
make clean
```
### 1.3 Tools

```shell
# Build Tools
make tool
# exec tool: edb
./tools/edb/edb
# clean build
make clean
```

## 2 Item Architecture

```
    [Inference System]

    ┌─────────┐┌─────────┐┌─────────┐
    │    C    ││   C++   ││  Python │
    └─────────┘└─────────┘└─────────┘     API
     ───────────────────────────────    ──────
    ┌─────────────────┐   ┌─────────┐ 
    │  Model Serial   │   │         │ 
    └─────────────────┘   │         │
    ┌─────────────────┐   │         │
    │  Graph IR Analy │   │ Monitor │
    └─────────────────┘   │         │
    ┌─────────────────┐   │         │
    │  Exec Schedule  │   │         │
    └─────────────────┘   └─────────┘   Engine
     ───────────────────────────────    ──────
    ┌─────────┐┌─────────┐┌─────────┐   Device
    │   CPU   ││   GPU   ││   NPU   │
    └─────────┘└─────────┘└─────────┘
```

### 2.1 Model File Serializer

Support model file format:
1. ONNX(*Open Neural Network Exchange*)
2. tflite(*TensorFlow Lite Model Format*)

### 2.2 Runtime Scheduler

1. Dynamic Batch
2. Heterogeneous Execute [Device]
3. Memory Alloca [Device]
4. Replica Parallelism [Device]

### 2.3 Kernel Operator Design


### 2.4 Edge Device Distribute


## Reference

- Libonnx
- Tengine
- MNN
- TFLM