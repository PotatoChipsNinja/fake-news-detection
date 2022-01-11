import torch
from torch import nn

class CNNExtractor(nn.Module):
    def __init__(self, feature_kernel, input_dim):
        super(CNNExtractor, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, feature_num, kernel_size) for kernel_size, feature_num in feature_kernel.items()])

    def forward(self, input):
        input = input.permute(0, 2, 1)
        feature = [conv(input) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]).squeeze() for f in feature]
        feature = torch.cat(feature, dim=1)
        return feature

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super(MLP, self).__init__()
        layers = list()
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
