
import numpy as np

import torch
from torch import nn

import gymnasium as gym
import ale_py

import cv2

from DQN.Qnet import AtariFramePreprocesser, AtariQnetwork

def test_AtariFramePreprocesser():
    env = gym.make("ALE/Breakout-v5")

    state, _ = env.reset()
    print(state.shape)

    cv2.imshow('hello', state)
    cv2.waitKey(0)

    afe = AtariFramePreprocesser(1, (84, 84), torch.device('cpu'))

    proced_frame = afe.preprocessing(np.array([[state, state]]))
    print(proced_frame.shape)

    frame = np.array(proced_frame[0][0])

    cv2.imshow('hello', frame)
    cv2.waitKey(0)

def test_AtariQnetwork():
    env = gym.make("ALE/Breakout-v5")

    state, _ = env.reset()
    print(state.shape)

    afe = AtariFramePreprocesser(1, (84, 84), torch.device('cpu'))

    proced_frame = afe.preprocessing(np.array([[state, state]]))

    model = AtariQnetwork((84, 84), 2, 4)

    ret = model.forward(proced_frame)

    print(ret)