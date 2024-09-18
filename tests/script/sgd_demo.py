import numpy as np

def sgd_update(parameters, gradients, learning_rate):
    """
    对模型参数进行一次SGD更新。
    :param parameters: 当前的模型参数，numpy数组。
    :param gradients: 对应于这些参数的梯度，numpy数组。
    :param learning_rate: 学习率，控制更新的步长。
    :return: 更新后的模型参数。
    """
    parameters -= learning_rate * gradients
    return parameters

# 示例: 线性回归模型 y = wx + b
# 损失函数：min((y - y')^2)
# 梯度：gead_w = 2x(wx + b - y)， grad_b = 2(wx + b - y)
w, b = 0.0, 0.0  # 初始化参数
learning_rate = 0.01  # 设置学习率

# 假设我们有一些训练数据
x_train = np.array([1, 2, 3, 4])
y_train = np.array([2, 4, 6, 8])

# 执行SGD优化
for epoch in range(100):  # 训练100轮
    for x, y in zip(x_train, y_train):
        # 计算当前参数下的梯度
        grad_w = 2 * x * (w * x + b - y)
        grad_b = 2 * (w * x + b - y)
        
        # 使用SGD更新参数
        w = sgd_update(w, grad_w, learning_rate)
        b = sgd_update(b, grad_b, learning_rate)

print(f"训练后的参数: w={w}, b={b}")