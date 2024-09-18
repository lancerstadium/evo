import torch
import torch.nn as nn
import torch.optim as optim
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

# 转换为PyTorch张量
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# 定义神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # 输入层2个神经元，隐含层1有3个神经元
        self.tanh = nn.Tanh()       # tanh激活函数
        self.fc2 = nn.Linear(3, 1)  # 输出层1个神经元，无激活函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 500
for epoch in range(epochs):
    model.train()
    
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    predicted = model(X_tensor)
    print("Predicted values: ", predicted.flatten().numpy())
