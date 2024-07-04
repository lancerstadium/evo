# evo


```
   $$$$$$\  $$\    $$\  $$$$$$\  
  $$  __$$\ \$$\  $$  |$$  __$$\ 
  $$$$$$$$ | \$$\$$  / $$ /  $$ |
  $$   ____|  \$$$  /  $$ |  $$ |
  \$$$$$$$\    \$  /   \$$$$$$  |
   \_______|    \_/     \______/ 

```


## 1 Build & Demo





## 2 Item Architecture

```
    [Inference System]

    ┌─────────────────┐   ┌─────────┐ 
    │  Model Serial   │   │         │ 
    └─────────────────┘   │         │
    ┌─────────────────┐   │         │
    │   Graph Analy   │   │ Monitor │
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

