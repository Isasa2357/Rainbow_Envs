
from typing import Tuple


import torch
from torch import nn

from mutil_RL.mutil_torch import factory_LinearReLU_Sequential

class Qnetwork(nn.Module):
    def __init__(self, in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int):
        super().__init__()
        
        self._network = factory_LinearReLU_Sequential(in_chnls, 64, 3, out_chnls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network.forward(x)
    
    def noise_reset(self):
        pass