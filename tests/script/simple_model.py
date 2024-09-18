import numpy as np

# 数据定义
X = np.array([
    [-3, -2],
    [-2.7, -1.8],
    [-2.4, -1.6],
    [-2.1, -1.4],
    [-1.8, -1.2],
    [-1.5, -1],
    [-1.2, -0.8],
    [-0.9, -0.6],
    [-0.6, -0.4],
    [-0.3, -0.2],
    [0, -2.22],
    [0.3, 0.2],
    [0.6, 0.4],
    [0.9, 0.6],
    [1.2, 0.8],
    [1.5, 1],
    [1.8, 1.2]
], dtype=np.float32)

y = np.array([
    0.6589, 0.2206, -0.1635, -0.4712, -0.6858, -0.7975,
    -0.804, -0.7113, -0.5326, -0.2875, 0, 0.3035,
    0.5966, 0.8553, 1.06, 1.1975, 1.2618
], dtype=np.float32).reshape(-1, 1)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ------------------- 随机梯度下降 -------------------
# 使用固定值初始化权重和偏置
W1 = np.array([[0.1, 0.2, 0.3], 
               [0.4, 0.5, 0.6]])  # (2, 3)
b1 = np.array([[0.01, 0.02, 0.03]])  # (1, 3)

W2 = np.array([[0.1], 
               [0.2], 
               [0.3]])  # (3, 1)
b2 = np.array([[0.05]])  # (1, 1)

learning_rate = 0.001
epochs = 10

loss_history_sgd = []

for epoch in range(epochs):
    loss_epoch = 0
    for i in range(X.shape[0]):
        xi = X[i:i+1]
        yi = y[i:i+1]

        # 前向传播
        Z1 = np.dot(xi, W1) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        y_pred = Z2

        # 计算损失
        loss = mean_squared_error(yi, y_pred)
        loss_epoch += loss

        # 反向传播
        dZ2 = y_pred - yi
        dW2 = np.dot(A1.T, dZ2)
        db2 = dZ2

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = np.dot(xi.T, dZ1)
        db1 = dZ1

        # 参数更新
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch == 0:
            print(Z1)
            print(A1)
            print(Z2)
            print('--')

    loss_epoch /= X.shape[0]
    loss_history_sgd.append(loss_epoch)

    # if (epoch + 1) % 300 == 0:
    #     print(f'SGD Epoch [{epoch+1}/{epochs}], Loss: {loss_epoch:.6f}')

# 预测
# Z1 = np.dot(X, W1) + b1
# A1 = tanh(Z1)
# Z2 = np.dot(A1, W2) + b2
# y_pred_sgd = Z2
# print("SGD Predicted values: ", y_pred_sgd.flatten())
# print("Actual values:        ", y.flatten())




# # ------------------- 批量梯度下降 -------------------
# # 初始化权重和偏置
# np.random.seed(0)
# W1 = np.random.randn(2, 3) * 0.01
# b1 = np.zeros((1, 3))

# W2 = np.random.randn(3, 1) * 0.01
# b2 = np.zeros((1, 1))

# learning_rate = 0.01
# epochs = 15000

# loss_history = []

# for epoch in range(epochs):
#     # 前向传播
#     Z1 = np.dot(X, W1) + b1
#     A1 = tanh(Z1)
#     Z2 = np.dot(A1, W2) + b2
#     y_pred = Z2

#     # 计算损失
#     loss = mean_squared_error(y, y_pred)
#     loss_history.append(loss)

#     # 反向传播
#     dZ2 = y_pred - y
#     dW2 = np.dot(A1.T, dZ2) / X.shape[0]
#     db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

#     dA1 = np.dot(dZ2, W2.T)
#     dZ1 = dA1 * tanh_derivative(Z1)
#     dW1 = np.dot(X.T, dZ1) / X.shape[0]
#     db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

#     # 参数更新
#     W2 -= learning_rate * dW2
#     b2 -= learning_rate * db2
#     W1 -= learning_rate * dW1
#     b1 -= learning_rate * db1

#     if (epoch + 1) % 300 == 0:
#         print(f'Batch GD Epoch [{epoch+1}/{epochs}], Loss: {loss:.6f}')

# # 预测
# Z1 = np.dot(X, W1) + b1
# A1 = tanh(Z1)
# Z2 = np.dot(A1, W2) + b2
# y_pred_batch = Z2
# print("Batch GD Predicted values: ", y_pred_batch.flatten())
# print("Actual values:             ", y.flatten())
