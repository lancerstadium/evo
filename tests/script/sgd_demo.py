import numpy as np  
  
# 目标函数（损失函数）和其梯度  
def loss_function(w, b, x, y):  
    return np.sum((y - (w * x + b)) ** 2) / len(x)  
  
def gradient_function(w, b, x, y):  
    dw = -2 * np.sum((y - (w * x + b)) * x) / len(x)  
    db = -2 * np.sum(y - (w * x + b)) / len(x)  
    return dw, db  
  
# SGD算法  
def sgd(x, y, learning_rate=0.01, epochs=1000):  
    # 初始化参数  
    w = np.random.rand()  
    b = np.random.rand()  
      
    # 存储每次迭代的损失值，用于可视化  
    losses = []  
      
    for i in range(epochs):  
        # 随机选择一个样本（在这个示例中，我们没有实际进行随机选择，而是使用了整个数据集。在大数据集上，你应该随机选择一个样本或小批量样本。）  
        # 注意：为了简化示例，这里我们实际上使用的是批量梯度下降。在真正的SGD中，你应该在这里随机选择一个样本。  
          
        # 计算梯度  
        dw, db = gradient_function(w, b, x, y)  
          
        # 更新参数  
        w = w - learning_rate * dw  
        b = b - learning_rate * db  
          
        # 记录损失值  
        loss = loss_function(w, b, x, y)  
        losses.append(loss)  
          
        # 每隔一段时间打印损失值（可选）  
        if i % 100 == 0:  
            print(f"Epoch {i}, Loss: {loss}")  
      
    return w, b, losses  
  
# 示例数据（你可以替换为自己的数据）  
x = np.array([1, 2, 3, 4, 5])  
y = np.array([2, 4, 6, 8, 10])  
  
# 运行SGD算法  
w, b, losses = sgd(x, y)  
print(f"Optimized parameters: w = {w}, b = {b}")
