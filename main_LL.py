
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch

import gymnasium as gym

from ReplayBuffer.Buffer import NstepSampleMaker, NstepReplayBuffer, ReplayBuffer
from usefulParam.Param import makeConstant
from DQN.Rainbow import RainbowAgent

def main():
    env = gym.make("LunarLander-v3")

    state_size = 8
    action_size = 1
    action_kinds = 4

    device = device=torch.device('cpu')
    replayBuf = ReplayBuffer(20000, state_size, action_size, action_type=torch.int, device=device)
    agent = RainbowAgent(makeConstant(0.99, device), makeConstant(0.005, device), 
                         state_size, action_size, action_kinds, 
                         (64, 64, 64), 0.5, "MSELoss", "Adam", 
                         replayBuf, 32, device)

    reward_history = list()
    episodes = 1000
    for episode in tqdm(range(episodes), position=0):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # アクションを選択
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            action = agent.get_action(state_tensor)
            action = action.cpu().detach().numpy()

            # 環境を進める
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = np.array(reward, dtype=np.float32)
            done = np.array(truncated or terminated, dtype=np.int8)

            # エージェントを更新
            agent.update(state, action, reward, next_state, done)

            # 後処理
            total_reward += reward
            state = next_state
        
        agent.noise_reset()
        
        reward_history.append(total_reward)
        tqdm.write(f"episode: {episode}, reward: {total_reward}")

    plt.plot(reward_history)
    plt.show()


if __name__ == '__main__':
    main()