import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class GumbelSoftmaxGate(nn.Module):
    def __init__(self, input_dim, num_gates):
        super(GumbelSoftmaxGate, self).__init__()
        self.fc = nn.Linear(input_dim, num_gates)
    
    def forward(self, x, tau=0.1):
        logits = self.fc(x)
        gate_probs = F.gumbel_softmax(logits, tau=tau, hard=True)
        return gate_probs

class MNISTWithGumbelGate(nn.Module):
    def __init__(self):
        super(MNISTWithGumbelGate, self).__init__()
        self.gate = GumbelSoftmaxGate(28*28, 3)  # 假设我们在三层之间进行门控
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展开为二维
        
        # 获取门控掩码
        gate_mask = self.gate(x)
        
        # 第一层
        if gate_mask[:, 0].mean() > 0.5:
            x = F.relu(self.fc1(x))
        
        # 第二层
        if gate_mask[:, 1].mean() > 0.5:
            x = F.relu(self.fc2(x))
        
        # 第三层
        if gate_mask[:, 2].mean() > 0.5:
            x = self.fc3(x)
        
        return x




if __name__ == "__main__":
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 创建模型
    model = MNISTWithGumbelGate()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(10):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

    # 测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test Accuracy: {correct / len(test_loader.dataset):.4f}')
