
import numpy as np
from tqdm import tqdm

import torch

import gymnasium as gym

from ReplayBuffer.Buffer import NstepSampleMaker
from usefulParam.Param import makeConstant

def main():
    env = gym.make("LunarLander-v3")

    state_size = 8
    action_size = 1
    action_kinds = 4

    maker = NstepSampleMaker(3, makeConstant(0.99), state_size, action_size)

    episodes = 2
    for episode in tqdm(range(episodes)):
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
            state = torch.tensor(state)
            action = torch.tensor(action)
            reward = torch.tensor(reward)
            next_state = torch.tensor(next_state)
            done = torch.tensor(done)
            print(done)
            maker.add(state, action, reward, next_state, done)
            if (maker.can_makeSample()):
                pass
                ret = maker.get_sample()
                print(ret)

            # 後処理
            total_reward += reward
            state = next_state

if __name__ == '__main__':
    main()