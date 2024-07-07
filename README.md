# evo


```
   $$$$$$\  $$\    $$\  $$$$$$\  
  $$  __$$\ \$$\  $$  |$$  __$$\ 
  $$$$$$$$ | \$$\$$  / $$ /  $$ |
  $$   ____|  \$$$  /  $$ |  $$ |
  \$$$$$$$\    \$  /   \$$$$$$  |
   \_______|    \_/     \______/ 

```

- Docs: [evo-docs](https://lancerstadium.github.io/evo/docs/)

## 1 Build & Test


### 1.1 Build

```shell
make
```

### 1.2 Test

```shell
# test all
make test
# test target unit
make test TEST_TRG=[xxx]_tests
```


## 2 Item Architecture

```
    [Inference System]

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

1. ONNX(*Open Neural Network Exchange*)

### 2.2 Runtime Scheduler

1. Dynamic Batch
2. Heterogeneous Execute [Device]
3. Memory Alloca [Device]
4. Replica Parallelism [Device]

### 2.3 Kernel Operator Design


### 2.4 Edge Device Distribute




## Reference

