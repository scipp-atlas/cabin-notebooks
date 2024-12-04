import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F

from LearningCutsUtils.OneToOneLinear import OneToOneLinear


class EfficiencyScanNetwork(torch.nn.Module):
    def __init__(self,features,effics,weights=None,activationscale=2.,postroot=1.):
        super().__init__()
        self.features = features
        self.effics = effics
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.post_product_root=postroot
        self.nets = torch.nn.ModuleList([OneToOneLinear(features, self.activation_scale_factor, self.weights, self.post_product_root) for i in range(len(self.effics))])

    def forward(self, x):
        outputs=torch.stack(tuple(self.nets[i](x) for i in range(len(self.effics))))
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)

