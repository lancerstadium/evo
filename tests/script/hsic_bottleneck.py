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
    
    def forward(self, x, y=None):
        z = x
        total_loss = 0.

        # During training, we compute the HSIC loss.
        if y is not None:
            m = x.size(0)
            Kx = kernel_matrix(x, self.sigma)
            Ky = kernel_matrix(y, self.sigma)
        
        for layer in self.layers:
            z = layer(z)
            if isinstance(layer, nn.Linear) and y is not None:
                Kz = kernel_matrix(z, self.sigma)
                loss = hsic(Kz, Kx, m) - self.lambda_0 * hsic(Kz, Ky, m)
                total_loss += loss

        out = self.output_layer(z)
        
        if y is not None:
            return out, total_loss
        else:
            return out  # Return only the output during evaluation or inference

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


def train_HSICBottleneckModel(train_loader, post_train=True, input_dim=25, hidden_dims=[40, 64, 32], output_dim=1, sigma=10., lambda_0=100., export_onnx=False, before_post_name="hsic_bottleneck_model_before_post_train", after_post_name="hsic_bottleneck_model_after_post_train"):
    # Define the model
    model = HSICBottleneckModel(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, sigma=sigma, lambda_0=lambda_0)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model with HSIC loss
    for epoch in range(50):
        model.train()
        total_loss = 0.
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, hsic_loss = model(inputs, targets.unsqueeze(1))
            loss = criterion(outputs, targets.unsqueeze(1)) + hsic_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader)}')

    # Save the model before post-training
    torch.save(model.state_dict(), before_post_name + ".pth")
    if export_onnx:
        export_to_onnx(model, input_dim, before_post_name)

    if not post_train:
        return model

    # Post-training
    post_trained_model = PostTrainedModel(model)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, post_trained_model.parameters()), lr=0.1)

    for epoch in range(10):
        post_trained_model.train()
        total_loss = 0.
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = post_trained_model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Fine-tuning Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader)}')
    
    # Save the model after post-training
    torch.save(post_trained_model.state_dict(), after_post_name + ".pth")
    if export_onnx:
        export_to_onnx(post_trained_model, input_dim, after_post_name)

    return model, post_trained_model

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
    # train_HSICBottleneckModel(train_loader, export_onnx=True)

    # Evaluate the model on the test set
    eval_HSICBottleneckModel("result/hsic_bottleneck_model_before_post_train.pth", test_loader)
    eval_HSICBottleneckModel("result/hsic_bottleneck_model_after_post_train.pth", test_loader, post_train=True)
