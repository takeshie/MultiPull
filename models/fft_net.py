import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import numpy as np
from functools import partial
from torch import nn
import copy

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, quantization_interval=2*np.pi):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        std = torch.sqrt(weight_scale / quantization_interval)
        self.linear.weight.data.normal_(-std, std)
        self.linear.bias.data.uniform_(-math.pi/4, math.pi/4)

    def forward(self, x):
        return torch.sin(self.linear(x))

class MFNBase(nn.Module):
    def __init__(self, hidden_size, out_size, n_layers, bias=False, output_act=False):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)])
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

    def forward(self, model_input):
        input_dict = {key: input.clone().detach().requires_grad_(True) for key, input in model_input.items()}
        coords = input_dict['coords']

        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)
        return {'model_in': input_dict, 'model_out': {'output': out}}

class FFT(MFNBase):
    def __init__(self, in_size=3, hidden_size=256, out_size=64, hidden_layers=8, bias=False, 
                 output_act=False, quantization_interval=np.pi):
        super().__init__(hidden_size, out_size, hidden_layers, bias, output_act)

        self.quantization_interval = quantization_interval
        frequency = torch.arange(1, hidden_layers + 1)
        weight_scale = torch.sqrt(torch.sin(math.pi * frequency / (hidden_layers + 1)))

        self.filters = nn.ModuleList([
            FourierLayer(in_size, hidden_size, scale, quantization_interval=quantization_interval)
            for scale in weight_scale
        ])
        self.output_linear = nn.ModuleList([nn.Linear(hidden_size, out_size) for _ in range(len(self.filters))])
        self.relu = nn.ReLU()
        self.reuse_filters = True

    def forward_mfn(self, input_dict):
        coords = input_dict
        outputs = []
        self.filter_outputs = [self.filters[0](coords)] * 2 + [self.filters[2](coords)] * 2 + \
                              [self.filters[4](coords)] * (len(self.filters) - 4)
        out = self.filter_outputs[0]
        for i in range(1, len(self.filters)):
            out = torch.nn.functional.normalize(self.filter_outputs[i] * self.relu(self.linear[i - 1](out)), p=2, dim=1)
            outputs.append(self.output_linear[i](out))
        return outputs

    def forward(self, model_input,index=-1):
        out = self.forward_mfn(model_input)
        init_list = [out[i] for i in [0, 2, 4]]
        init_feat = torch.cat(init_list, dim=1)
        indices = [5, 6, -1]
        res_out = [torch.nn.functional.normalize(torch.cat((out[idx], init_feat), dim=1), p=2, dim=1) for idx in indices]
        return res_out[index]