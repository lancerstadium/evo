import numpy as np
import struct

# 1. 读取MNIST数据集的函数
def load_mnist_images(filename):
    """读取MNIST图像文件并返回一个形状为 (num_images, 28*28) 的 numpy 数组"""
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))  # 读取文件头
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows * cols)  # 读取图像数据并展平
    return images / 255.0  # 正则化，将像素值从 0-255 转换为 0-1

def load_mnist_labels(filename):
    """读取MNIST标签文件并返回一个形状为 (num_labels,) 的 numpy 数组"""
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))  # 读取文件头
        labels = np.fromfile(f, dtype=np.uint8)  # 读取标签数据
    return labels

def one_hot_encode(labels, num_classes=10):
    """将标签转换为 one-hot 编码"""
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# 2. 定义网络参数
np.random.seed(42)  # 固定随机种子以确保每次结果一致
input_size = 28 * 28  # MNIST图片展平后的尺寸 28x28
hidden_size = 128     # 隐藏层大小
output_size = 10      # 输出层大小（分类为10类）

# 初始化权重和偏置
W1 = np.random.randn(input_size, hidden_size) * 0.01  # 输入到隐藏层权重
b1 = np.zeros((1, hidden_size))  # 隐藏层偏置
W2 = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏层到输出层权重
b2 = np.zeros((1, output_size))  # 输出层偏置

# 3. 激活函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数值稳定性调整
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross entropy loss
def cross_entropy_loss(y_true, y_pred):
    loss = -np.sum(y_true * np.log(y_pred + 1e-10))  # 加上 1e-10 防止 log(0)
    return loss / y_true.shape[0]

# 4. 前向传播
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# 5. 反向传播
def backward(X, y_true, z1, a1, z2, a2, learning_rate=0.01):
    global W1, b1, W2, b2
    
    # 输出层到隐藏层的梯度
    delta2 = (a2 - y_true)  # 使用交叉熵损失的梯度
    dW2 = np.dot(a1.T, delta2)  # 隐藏层权重的梯度
    db2 = np.sum(delta2, axis=0, keepdims=True)  # 输出层偏置的梯度
    
    # 隐藏层到输入层的梯度
    delta1 = np.dot(delta2, W2.T) * tanh_derivative(z1)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)
    
    # 参数更新 (SGD)
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# 6. 训练过程
def train(X_train, y_train, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        loss = 0
        for i in range(X_train.shape[0]):  # 使用每个样本进行SGD
            X = X_train[i:i+1]  # 单个样本
            y = y_train[i:i+1]  # 对应的标签
            
            # 前向传播
            z1, a1, z2, a2 = forward(X)
            
            # 计算损失
            loss += cross_entropy_loss(y, a2)
            
            # 反向传播和参数更新
            backward(X, y, z1, a1, z2, a2, learning_rate)
        
        loss /= X_train.shape[0]  # 平均损失
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

# 7. 预测
def predict(X):
    _, _, _, a2 = forward(X)
    return np.argmax(a2, axis=1)

# 8. 读入MNIST数据集
X_train = load_mnist_images('../picture/mnist/train-images-idx3-ubyte')  # 训练集图像
y_train = load_mnist_labels('../picture/mnist/train-labels-idx1-ubyte')  # 训练集标签
y_train = one_hot_encode(y_train)  # 将标签转换为one-hot编码

X_test = load_mnist_images('../picture/mnist/t10k-images-idx3-ubyte')  # 测试集图像
y_test = load_mnist_labels('../picture/mnist/t10k-labels-idx1-ubyte')  # 测试集标签
y_test = one_hot_encode(y_test)  # 将标签转换为one-hot编码

# 打印数据集形状进行检查
print("Training set shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test set shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# 9. 训练模型
train(X_train, y_train, epochs=10, learning_rate=0.01)

# 10. 使用训练后的模型进行预测
predictions = predict(X_test[:10])  # 预测前10个测试样本
print("Predictions:", predictions)
