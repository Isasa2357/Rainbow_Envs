
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
    # env = gym.make("CartPole-v1")

    state_size = 8
    action_size = 1
    action_kinds = 4

    device =torch.device('cpu')
    replayBuf = ReplayBuffer(20000, state_size, action_size, action_type=torch.int, device=device)
    agent = RainbowAgent(makeConstant(0.99, device), makeConstant(1e-4, device), makeConstant(5e-3, device), 
                         state_size, action_size, action_kinds, 
                         (64, 64, 64), 0.3, "MSELoss", "Adam", 1, 
                         replayBuf, 64, device)
    
    # あらかじめ適当な量バッファを埋めておく
    bufinit_episodes = 100
    for episode in tqdm(range(bufinit_episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = np.random.choice(range(action_kinds))
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = np.array(reward, dtype=np.float32)
            done = np.array(truncated or terminated, dtype=np.int8)

            agent.add_buffer(state, action, reward, next_state, done)

            state = next_state

            agent.noise_reset()

    reward_history = list()
    episodes = 3000000000000
    # episodes = 30
    over_200_count = 0
    for episode in tqdm(range(episodes), position=0, ncols=100):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        action_history = [0] * action_kinds

        while not done:
            # アクションを選択
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            action = agent.get_action(state_tensor)
            action = action.cpu().detach().numpy()

            action_history[int(action.item())] += 1

            # 環境を進める
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = np.array(reward, dtype=np.float32)
            done = np.array(truncated or terminated, dtype=np.int8)

            # エージェントを更新
            agent.update(state, action, reward, next_state, done)

            # 後処理
            total_reward += reward
            state = next_state
        
        reward_history.append(total_reward if total_reward > -2000.0 else -2000.0)
        tqdm.write(f"episode: {episode}, reward: {total_reward}")
        tqdm.write(f'action history: {action_history}')

        if total_reward >= 200:
            over_200_count += 1
        else:
            over_200_count = 0
        
        if over_200_count >= 100:
            print("clear")
            break

    plt.plot(reward_history)
    plt.show()


if __name__ == '__main__':
    main()