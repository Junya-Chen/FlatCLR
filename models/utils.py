import torch
import torch.nn as nn


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x  
    
    
class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size, h_dim=[], use_bn=False):
        super(MLPLayer, self).__init__()
        net = []
        hs = [input_size] + h_dim + [output_size]
        if use_bn:
            for h0, h1 in zip(hs, hs[1:]):
                net.extend([
                    nn.Linear(h0, h1),
                    nn.BatchNorm1d(h1),
                    nn.ReLU()])
        else:
            for h0, h1 in zip(hs, hs[1:]):
                net.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU()])
        net.pop()  # pop the last ReLU for the output layer.
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        z = self.net(x)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, args, in_channel):
        super(ProjectionHead, self).__init__()
        self.args = args
        out_dim = self.args.proj_out_dim
        self.linear_layers = []
    
        if self.args.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
    
        elif self.args.proj_head_mode == 'linear':
            h_dim = []
            self.linear_layers = MLPLayer(in_channel, out_dim, h_dim, use_bn=True)
            
        elif self.args.proj_head_mode =='nonlinear':
            h_dim = [in_channel]*(self.args.num_proj_layers - 1)
            self.linear_layers = MLPLayer(in_channel, out_dim, h_dim, use_bn=False)
        else:
            raise ValueError('Unknown head projection mode {}'.format(self.args.proj_head_mode))
    
    def forward(self, inputs):
        if self.args.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        
        elif self.args.proj_head_mode == 'linear' or self.args.proj_head_mode == 'nonlinear':
            proj_input = inputs
            proj_head_output = self.linear_layers(inputs)
            return proj_input, proj_head_output
        else:
            raise ValueError('Unknown head projection mode {}'.format(self.args.proj_head_mode))
            
    
class SupervisedHead(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(SupervisedHead, self).__init__()
        self.linear_layer = MLPLayer(in_channel, num_classes)
    
    def forward(self, inputs):
        pred = self.linear_layer(inputs)
        return pred