import torch
import torch.nn.functional as F
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

# Define the PyTorch model
class GumbelSoftmaxModel(torch.nn.Module):
    def __init__(self):
        super(GumbelSoftmaxModel, self).__init__()

    def forward(self, h):
        return gumbel_softmax_sampling(h)

def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.1):
    p = F.softmax(h, dim=1)
    y = torch.rand(h.shape, dtype=torch.float32) + 1e-25
    gumbels = mu - beta * torch.log(-torch.log(y))
    x = (torch.log(p) + gumbels) / tau
    return F.softmax(x, dim=1)

# Test the model and export to ONNX
model = GumbelSoftmaxModel()
h = torch.rand((10, 3))  # Example input tensor
out = model(h)
out_ref = F.gumbel_softmax(h, tau=0.1)
print(out)
print(out_ref)

torch.onnx.export(model, h, 'gumbel_softmax.onnx', opset_version=9)
