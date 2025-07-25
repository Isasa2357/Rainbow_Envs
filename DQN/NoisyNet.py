'''
NoisyNet


'''

import torch
from torch import nn

from typing import Tuple, cast
import inspect

from mutil_RL.mutil_torch import factory_LinearReLU_Sequential
from mtorch.NoisyLinear import factory_NoisyLiearReLU_Sequential, NoisyLinear
from DQN.Qnet import BaseQnetwork

class NoisyNetInterface(BaseQnetwork):
    def __init__(self):
        super().__init__()

    def noise_reset(self):
        raise NotImplementedError(f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}は未実装') # type: ignore
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}は未実装') # type: ignore


class NoisyNet(NoisyNetInterface):
    '''
    Noisy Net
    '''

    def __init__(self, in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int, sigma_init: float=0.5):
        super().__init__()
        
        self._network = factory_NoisyLiearReLU_Sequential(in_chnls, hdn_chnls, out_chnls, tuple([sigma_init] * (len(hdn_chnls) + 1)))
    
    def noise_reset(self):
        '''
        NoisyLinearのノイズのリセット
        '''
        for i in range(len(self._network)):
            if isinstance(self._network[i], NoisyLinear):
                noisy_layer = cast(NoisyLinear, self._network[i])
                noisy_layer.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self._network.forward(x)

class DuelingNoisyNet(NoisyNetInterface):
    '''
    Dueling Noisy Net
    '''

    def __init__(self, in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int, sigma_init: float=0.5):
        super().__init__()

        # duelingNet構成用にhdnchnlを修正
        hdn_chnls_4factory = hdn_chnls[0:len(hdn_chnls) - 1]
        out_chnls_4factory = hdn_chnls[-1]
        
        self._comm_network = factory_NoisyLiearReLU_Sequential(in_chnls, hdn_chnls_4factory, out_chnls_4factory, tuple([sigma_init] * (len(hdn_chnls_4factory) + 1)))
        self._comm_network.append(nn.ReLU())

        self._base_layer = NoisyLinear(out_chnls_4factory, 1, sigma_init)
        self._advantage_layer = NoisyLinear(out_chnls_4factory, out_chnls, sigma_init)
    
    def noise_reset(self):
        '''
        NoisyLinearのノイズのリセット
        '''
        for i in range(len(self._comm_network)):
            if isinstance(self._comm_network[i], NoisyLinear):
                noisy_layer = cast(NoisyLinear, self._comm_network[i])
                noisy_layer.reset_noise()
        
        self._base_layer.reset_noise()
        self._advantage_layer.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._comm_network.forward(x)
        
        base = self._base_layer.forward(x)
        advantage = self._advantage_layer.forward(x)

        return base + advantage - torch.mean(advantage, dim=1, keepdim=True)

############################## テスト ##############################

def test_NoisyNet():
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int)

    net = NoisyNet(4, (32, 64, 32), 4, 0.5)

    print(net)

    print(net.forward(x))

    print(net.forward(x))

    net.noise_reset()

    print(net.forward(x))

def test_DuelingNoisyNet():
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int)

    net = DuelingNoisyNet(4, (32, 64, 32), 4, 0.5)

    print(net)

    print(net.forward(x))

    print(net.forward(x))

    net.noise_reset()

    print(net.forward(x))

def test_commDefine():
    def net_forward(net: NoisyNetInterface, x: torch.Tensor) -> torch.Tensor:
        return net.forward(x)
    
    net = NoisyNet(4, (32, 64, 32), 4, 0.5)
    dnet = NoisyNet(4, (32, 64, 32), 4, 0.5)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int)

    print(net_forward(net, x))
    print(net_forward(dnet, x))
