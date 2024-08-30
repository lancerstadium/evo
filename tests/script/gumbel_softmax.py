import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

# Define the symbolic function
@parse_args('v', 'f', 'f', 'f')
def gumbel_softmax_symbolic(g, h, mu, beta, tau):
    p = g.op("Softmax", h, axis_i=1)
    
    # Generate random noise in the range (0, 1) and apply inverse Gumbel CDF
    y = g.op("RandomUniformLike", h, dtype_i=1, low_f=1e-25, high_f=1.0)
    gumbels = g.op("Sub", g.op("Mul", g.op("Neg", g.op("Log", g.op("Neg", g.op("Log", y)))), g.op("Constant", value_t=torch.tensor(beta, dtype=torch.float32))), g.op("Constant", value_t=torch.tensor(mu, dtype=torch.float32)))

    # Add the Gumbel noise to log probabilities and scale by temperature tau
    x = g.op("Add", g.op("Log", p), gumbels)
    x = g.op("Div", x, g.op("Constant", value_t=torch.tensor(tau, dtype=torch.float32)))

    # Apply softmax to the resulting tensor
    return g.op("Softmax", x, axis_i=1)

# Register the custom symbolic function
register_custom_op_symbolic('::gumbel_softmax', gumbel_softmax_symbolic, 9)


class GumbelSoftmaxModule(torch.nn.Module):
    def __init__(self, tau=1, hard=False, eps=1e-10, dim=-1):
        super(GumbelSoftmaxModule, self).__init__()
        self.tau = tau
        self.hard = hard
        self.eps = eps
        self.dim = dim

    def sample_gumbel(self, shape, device, eps=1e-10):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax(self, logits):
        gumbel_noise = self.sample_gumbel(logits.size(), logits.device, self.eps)
        y = logits + gumbel_noise
        
        # Softmax操作
        y_soft = F.softmax(y / self.tau, dim=self.dim)
        
        if self.hard:
            # 将softmax结果转换为one-hot
            index = y_soft.max(self.dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(self.dim, index, 1.0)
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft
        return y
    
    def forward(self, logits):
        return self.gumbel_softmax(logits)


# Test the model and export to ONNX
hard    = True
tau     = 1
logits  = Variable(torch.rand((10, 3)))  # Example input tensor
model   = GumbelSoftmaxModule(tau=tau, hard=hard)
name    = f"gumbel_softmax_{hard}.onnx"
out     = model(logits)
out_ref = F.gumbel_softmax(logits, tau=tau, hard=hard)
print(out)
print(out_ref)

torch.onnx.export(
    model,                          # 要导出的模型
    logits,                         # 模型的输入
    name,                           # 导出的ONNX文件名
    opset_version=11,               # ONNX opset版本
    input_names=['input'],          # 输入节点的名称
    output_names=['output'],        # 输出节点的名称
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'category_size'},  # 指定第二个维度为动态维度
        'output': {0: 'batch_size'}                     # 输出的第一个维度为动态
    }                
)
