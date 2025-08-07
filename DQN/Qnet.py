
from typing import Tuple, List

import numpy as np
from numpy import ndarray

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

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
    
class DuelingQnetwork(BaseQnetwork):
    def __init__(self, in_chnls: int, hdn_chnls: List[int], out_chnls: int):
        super().__init__()

        in_chnls_forFactory = in_chnls
        hdn_chnls_forFactory = tuple(hdn_chnls[:-1])
        out_chnls_forFactory = hdn_chnls[-1]

        self._comm_network = factory_LinearReLU_Sequential(in_chnls_forFactory, hdn_chnls_forFactory, out_chnls, out_act="ReLU")

        self._base_layer = nn.Linear(out_chnls_forFactory, 1)
        self._advantage_layer = nn.Linear(out_chnls_forFactory, out_chnls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: [N, in_chnls]
        
        Rets:
            out: [N, out_chnls]
        '''

        x = self._comm_network.forward(x)

        base = self._base_layer.forward(x)
        advantage = self._base_layer.forward(x)

        return base + advantage

class AtariFramePreprocesser:
    '''
    Atariのフレームを強化学習用に変換するクラス
    '''

    def __init__(self, frame_num: int, out_size: Tuple[int, int], device: torch.device):
        self._frame_num = frame_num
        self._out_size = out_size  # (height, width)
        self._device = device
    
    def preprocessing(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: np.ndarray of shape [N, frame_num, H, W, 3], dtype uint8, in [0,255]
        
        Returns:
            torch.Tensor of shape [N, frame_num, out_h, out_w], dtype float32, normalized to [0,1]
        """
        # 1) NumPy → CUDA Tensor, reshape to [N*F, 3, H, W]
        batch_size, frame_num, height, width, channel = frames.shape
        x = torch.from_numpy(frames).to(self._device)                    # [N,F,H,W,3], uint8
        x = x.view(batch_size * frame_num, height, width, channel).permute(0, 3, 1, 2).float() / 255.0  # [N*F,3,H,W], float32
        
        # 2) Crop out “上下のいらない部分” (例: row 0–33, 194–end をカット)
        #    ここでは height 次元を 34:194 の範囲に限定
        x = x[:, :, 34:194, :]  # → [N*F,3,160,160]  (例: 210→160)

        # 3) RGB → グレースケール
        #    (0.2989 R + 0.5870 G + 0.1140 B)
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [N*F,1,H',W']

        # 4) リサイズ（双一次補間）
        #    out_size は (out_h, out_w)
        gray_resized = F.interpolate(
            gray,               # GPU 上の Tensor
            size=self._out_size,
            mode='bilinear',
            align_corners=False
        ) # [N*F,1,out_h,out_w]

        # 5) 最終形状に整形して返却
        out_h, out_w = self._out_size
        ret = gray_resized.view(batch_size, frame_num, out_h, out_w)  # [N,frame_num,out_h,out_w]
        return ret

class AtariQnetwork(BaseQnetwork):
    '''
    画像を入力とするQnetwork
    '''
    def __init__(self, frame_size: Tuple[int, int], frame_num: int, out_chnls: int):
        super().__init__()

        self._frame_size = frame_size
        self._frame_num = frame_num
        self._out_chnls = out_chnls

        self._fe_fromImg = nn.Sequential(
            nn.Conv2d(self._frame_num, 32, kernel_size=8, stride=4, padding=0), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
        )

        self._fe = nn.Linear(64*7*7, 512)
        self._fe_relu = nn.ReLU()
        self._head = nn.Linear(512, self._out_chnls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: torch.Tensor of shape [N, frame_num, height, width]
        
        Ret:
            torch.Tenosr of shape [N, action_size]
        '''

        x = self._fe_fromImg.forward(x)
        x = x.view(x.size(0), -1)
        x = self._fe.forward(x)
        x = self._fe_relu.forward(x)
        print(x.shape)
        x = self._head.forward(x)
        return x