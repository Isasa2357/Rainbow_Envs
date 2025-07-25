
from typing import Tuple


import torch
from torch import nn

from mutil_RL.mutil_torch import factory_LinearReLU_Sequential

class BaseQnetwork(nn.Module):
    '''
    作成したQ networkが共通して持つ属性
    '''
    def __init__(self):
        super().__init__()

class Qnetwork(BaseQnetwork):
    def __init__(self, in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int):
        super().__init__()
        
        self._network = factory_LinearReLU_Sequential(in_chnls, hdn_chnls, out_chnls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network.forward(x)
    
# class DuelingQnetwork(BaseQnetwork):
#     def __init__(self, in_chnls)