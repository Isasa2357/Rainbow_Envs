'''
    主にNoisy NetのNoisy Linear用に作成
'''

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple

class NoisyLinear(nn.Module):
    
    def __init__(self, in_chnls: int, out_chnls: int, sigma_init: float=0.5):
        super().__init__()

        self._in_chnls = in_chnls
        self._out_chnls = out_chnls

        # weight
        self._weight_mu = nn.Parameter(torch.empty(self._out_chnls, self._in_chnls))
        self._weight_sigma = nn.Parameter(torch.empty(self._out_chnls, self._in_chnls))
        self._weight_epsilon: torch.Tensor  # エディタの型推論用
        self.register_buffer("_weight_epsilon", torch.empty(out_chnls, in_chnls))

        # bias
        self._bias_mu = nn.Parameter(torch.empty(self._out_chnls))
        self._bias_sigma = nn.Parameter(torch.empty(self._out_chnls))
        self._bias_epsilon: torch.Tensor    # エディタの型推論用
        self.register_buffer("_bias_epsilon", torch.empty(self._out_chnls))

        self._sigma_init = sigma_init

        self._reset_parameter()
        self.reset_noise()
    

    def _reset_parameter(self):
        '''
        He初期化(ガウス分布)
        '''
        mu_init_sigma = (2.0 / float(self._in_chnls))**0.5

        # weight
        self._weight_mu.data.normal_(0.0, mu_init_sigma)
        self._weight_sigma.data.fill_(self._sigma_init / (self._in_chnls**0.5))

        # bias
        self._bias_mu.data.normal_(0.0, mu_init_sigma)
        self._bias_sigma.data.fill_(self._sigma_init / (self._in_chnls**0.5))

    def _scale_noise(self, size: int) -> torch.Tensor:

        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        '''
        探索率の再設定
        '''
        eps_in = self._scale_noise(self._in_chnls)
        eps_out = self._scale_noise(self._out_chnls)

        self._weight_epsilon.copy_(eps_out.ger(eps_in))
        self._bias_epsilon.copy_(eps_out)
    
    def __str__(self):
        return f"NoisyLinear({self._in_chnls}, {self._out_chnls}, {self._sigma_init})"
    
    def __repr__(self):
        return f"NoisyLinear(in_chnls={self._in_chnls}, out_chnls={self._out_chnls}, sigma_init={self._sigma_init})"
    
    def extra_repr(self):
        return f"in_chnls={self._in_chnls}, out_chnls={self._out_chnls}, sigma_init={self._sigma_init}"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        weight: torch.Tensor
        bias: torch.Tensor
        if self.training:
            weight = self._weight_mu + self._weight_sigma * self._weight_epsilon
            bias = self._bias_mu + self._bias_sigma * self._bias_epsilon
        else:
            weight = self._weight_mu
            bias = self._bias_mu
        return F.linear(x, weight, bias)


def factory_NoisyLiearReLU_Sequential(in_chnls: int, hdn_chnls: Tuple[int, ...],out_chnls: int, sigma_inits: Tuple[float, ...]) -> nn.Sequential:
    chnls = tuple([in_chnls]) + hdn_chnls + tuple([out_chnls])

    layers = list()

    for i in range(len(chnls) - 1):
        layers.append(NoisyLinear(chnls[i], chnls[i + 1], sigma_inits[i]))

        if i != len(chnls) - 2:
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

############################## テスト用 ##############################

def test_factory_NosiyReLu_Sequential():
    print(factory_NoisyLiearReLU_Sequential(8, (64, 128, 64), 4, (0.5, 0.5, 0.5, 0.5, 0.5)))

if __name__ == '__main__':
    test_factory_NosiyReLu_Sequential()
