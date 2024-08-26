import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Kernel Matrix Calculation
def kernel_matrix(x, sigma):
    x1 = x.unsqueeze(0)
    x2 = x.unsqueeze(1)
    return torch.exp(-0.5 * torch.sum((x1 - x2) ** 2, dim=-1) / sigma ** 2)

# HSIC Loss Calculation
def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

class HSICBottleneckModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, sigma, lambda_0):
        super(HSICBottleneckModel, self).__init__()
        self.sigma = sigma
        self.lambda_0 = lambda_0
        self.layers = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.ReLU())
            if i == 1:  # Add Dropout after second layer as in original Keras model
                self.layers.append(nn.Dropout(0.2))
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
        return self.output_layer(z)
    
    def compute_hsic_loss(self, x, y):
        m = x.size(0)
        Kx = kernel_matrix(x, self.sigma)
        Ky = kernel_matrix(y, self.sigma)
        
        z = x
        total_hsic_loss = 0.
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                z = layer(z)
                Kz = kernel_matrix(z, self.sigma)
                total_hsic_loss += hsic(Kz, Kx, m) - self.lambda_0 * hsic(Kz, Ky, m)

        return total_hsic_loss

class PostTrainedModel(nn.Module):
    def __init__(self, model):
        super(PostTrainedModel, self).__init__()
        self.model = model
        for name, param in self.model.named_parameters():
            if 'output_layer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        return self.model(x, x)[0]  # Pass x for y as dummy, since we only care about forward pass

# Function to generate data
def generate_data(num_samples=256*400, input_dim=25):
    X = np.random.standard_normal((num_samples, input_dim)).astype(np.float32)
    y = np.uint8(np.sum(X ** 2, axis=-1) > 25.).astype(np.float32)
    return X, y


def load_model(model_class, model_path, post_train=False, input_dim=25, hidden_dims=[40, 64, 32], output_dim=1, sigma=10., lambda_0=100.):
    # Instantiate the base model
    base_model = model_class(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, sigma=sigma, lambda_0=lambda_0)
    
    # Load the model weights
    state_dict = torch.load(model_path)
    
    if post_train:
        # Wrap the base model in PostTrainedModel
        model = PostTrainedModel(base_model)
        # Check if the keys already have "model." prefix
        new_state_dict = {}
        for key in state_dict.keys():
            if key.startswith("model."):
                # If keys already have "model." prefix, directly use them
                new_state_dict[key] = state_dict[key]
            else:
                # Otherwise, add the "model." prefix
                new_key = f"model.{key}"
                new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    else:
        model = base_model
        model.load_state_dict(state_dict)
    
    return model

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)  # Only inputs are needed during evaluation
            predicted = torch.round(torch.sigmoid(outputs))
            total += targets.size(0)
            correct += (predicted.view(-1) == targets).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')


def export_to_onnx(model, input_dim, name):
    # Create a dummy input with the correct shape
    dummy_input = torch.randn(1, input_dim)
    
    # Export the model
    torch.onnx.export(model, dummy_input, name + ".onnx", export_params=True, opset_version=11, 
                      do_constant_folding=True, input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Model has been exported to ONNX format: " + name + ".onnx")


def adaptive_perturbation(sigma, previous_loss, current_loss, perturbation_rate):
    """自适应调整扰动大小"""
    if current_loss < previous_loss:
        sigma *= 1 - perturbation_rate
    else:
        sigma *= 1 + perturbation_rate
    return sigma

def perturb_params(model, sigma=0.01):
    """随机扰动模型参数"""
    perturbed_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            perturbed_params[name] = param + sigma * torch.randn_like(param)
    return perturbed_params

def forward_forward_training(model, train_loader, epochs=50, lr=0.001, sigma=0.01, perturbation_rate=0.1, freeze_threshold=1e-3):
    frozen_params = set()
    for epoch in range(epochs):
        total_positive_loss = 0.0
        total_negative_loss = 0.0

        for inputs, targets in train_loader:
            # 计算当前参数下的正样本 HSIC 损失
            positive_hsic_loss = model.compute_hsic_loss(inputs, targets.unsqueeze(1))

            # 生成负样本
            negative_inputs = torch.randn_like(inputs)
            negative_targets = torch.zeros_like(targets)
            negative_hsic_loss = model.compute_hsic_loss(negative_inputs, negative_targets.unsqueeze(1))

            # 克隆原始参数
            original_params = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad and name not in frozen_params}

            # 对部分参数进行扰动
            perturbed_params = perturb_params(model, sigma)
            
            # 应用扰动后的参数
            for name, param in model.named_parameters():
                if param.requires_grad and name in perturbed_params:
                    param.data = perturbed_params[name].data
            
            # 重新计算扰动后的正负样本 HSIC 损失
            perturbed_positive_loss = model.compute_hsic_loss(inputs, targets.unsqueeze(1))
            perturbed_negative_loss = model.compute_hsic_loss(negative_inputs, negative_targets.unsqueeze(1))

            # 自适应调整扰动幅度
            sigma = adaptive_perturbation(sigma, positive_hsic_loss.item(), perturbed_positive_loss.item(), perturbation_rate=perturbation_rate)

            # 决定是否保留扰动
            update_needed = False
            for name, param in model.named_parameters():
                if param.requires_grad and name in original_params:
                    if perturbed_positive_loss < positive_hsic_loss or perturbed_negative_loss > negative_hsic_loss:
                        update_needed = True
                        if torch.abs(param - original_params[name]).max() < freeze_threshold:
                            frozen_params.add(name)  # 冻结参数
                    else:
                        param.data = original_params[name].data  # 回滚参数

            # 如果无需更新，则跳出当前 batch 的训练
            if not update_needed:
                break
            
            total_positive_loss += positive_hsic_loss.item()
            total_negative_loss += negative_hsic_loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Positive Loss: {total_positive_loss/len(train_loader):.8f}, Negative Loss: {total_negative_loss/len(train_loader):.8f}, Frozen Parameters: {len(frozen_params)}/{len(list(model.named_parameters()))}')


def train_HSICBottleneckModel_ff(train_loader, post_train=True, input_dim=25, hidden_dims=[40, 64, 32], output_dim=1, sigma=10., lambda_0=100., export_onnx=False, before_post_name="hsic_bottleneck_model_before_post_train_ff", after_post_name="hsic_bottleneck_model_after_post_train_ff"):
    # Define the model
    model = HSICBottleneckModel(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, sigma=sigma, lambda_0=lambda_0)
    torch.save(model.state_dict(), before_post_name + "_origin.pth")
    # forward forward train
    forward_forward_training(model, train_loader, epochs=50, lr=0.001, sigma=0.01)
    # Save the model before post-training
    torch.save(model.state_dict(), before_post_name + ".pth")
    if export_onnx:
        export_to_onnx(model, input_dim, before_post_name)

    # if not post_train:
    return model

    # Post-training
    # post_trained_model = PostTrainedModel(model)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, post_trained_model.parameters()), lr=0.1)

    # for epoch in range(10):
    #     post_trained_model.train()
    #     total_loss = 0.
    #     for inputs, targets in train_loader:
    #         optimizer.zero_grad()
    #         outputs = post_trained_model(inputs)
    #         loss = criterion(outputs, targets.unsqueeze(1))
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     print(f'Fine-tuning Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader)}')
    
    # # Save the model after post-training
    # torch.save(post_trained_model.state_dict(), after_post_name + ".pth")
    # if export_onnx:
    #     export_to_onnx(post_trained_model, input_dim, after_post_name)
    # return model, post_trained_model

def eval_HSICBottleneckModel(model_path, test_loader, post_train=False):
    if not post_train:
        print(f"Evaluating model before post-training: {model_path}")
        model_before_post_train = load_model(HSICBottleneckModel, model_path)
        evaluate_model(model_before_post_train, test_loader)
        return model_before_post_train
    else:
        print(f"Evaluating model after post-training: {model_path}")
        post_trained_model = load_model(HSICBottleneckModel, model_path, post_train=True)
        evaluate_model(post_trained_model, test_loader)
        return post_trained_model


# Main testing and training loop
if __name__ == "__main__":
    import numpy as np

    # Generate data
    X, y = generate_data()

    # Convert to PyTorch tensors
    X = torch.tensor(X)
    y = torch.tensor(y)

    # Split data into training and testing
    num_train = 256 * 360
    X_train, X_test = X[:num_train], X[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Training
    train_HSICBottleneckModel_ff(train_loader, export_onnx=True)

    # Evaluate the model on the test set
    eval_HSICBottleneckModel("hsic_bottleneck_model_before_post_train_ff_origin.pth", test_loader)
    eval_HSICBottleneckModel("hsic_bottleneck_model_before_post_train_ff.pth", test_loader)
