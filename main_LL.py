
import numpy as np
from tqdm import tqdm

import torch

import gymnasium as gym

from ReplayBuffer.Buffer import NstepSampleMaker, NstepReplayBuffer
from usefulParam.Param import makeConstant

def main():
    env = gym.make("LunarLander-v3")

    state_size = 8
    action_size = 1
    action_kinds = 4

    maker = NstepSampleMaker(3, makeConstant(0.99), state_size, action_size)
    buf = NstepReplayBuffer(20000, 7, makeConstant(0.99), state_size, action_size)

    episodes = 2
    for episode in tqdm(range(episodes), position=0):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # アクションを選択
            action = np.random.choice(range(action_kinds), 1)

            # 環境を進める
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = np.array(reward, dtype=np.float32)
            done = np.array(truncated or terminated, dtype=np.int8)

            # エージェントを更新
            buf.add(state, action, reward, next_state, done)

            # 後処理
            total_reward += reward
            state = next_state

            if buf.real_size >= 2:
                status, actions, rewrads, next_status, dones = buf.get_sample(2)
                print(f'status: {status}')
                print(f'action: {actions}')
                print(f'reward: {rewrads}')
                print(f'n state: {next_status}')
                print(f'done: {dones}')
                print()
        # tqdm.write(f"episode: {episode}, reward: {total_reward}")

if __name__ == '__main__':
    