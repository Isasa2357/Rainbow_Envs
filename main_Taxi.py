
from typing import Tuple

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import deque

import torch
from torch.nn import functional as F

import gymnasium as gym
from gymnasium import Env


from mutil_RL.mutil_gym import get_env_info
from ReplayBuffer.Buffer_v2 import ReplayBuffer, NstepReplayBuffer, PERBuffer, NstepPERBuffer
from DQN.Rainbow import RainbowAgent, warmup_Rainbow
from usefulParam.Param import makeConstant, makeMultiply

def decode_taxi_state(state):
    destination = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    return taxi_row, taxi_col, passenger, destination

def conv_taxi_state2onehots(state: int) -> torch.Tensor:
    taxi_row, taxi_col, passenger, destination = decode_taxi_state(state)
    taxi_row_onehot = F.one_hot(torch.tensor(taxi_row), 5)
    taxi_col_onehot = F.one_hot(torch.tensor(taxi_col), 5)
    passenger_onehot = F.one_hot(torch.tensor(passenger), 5)
    destination_onehot = F.one_hot(torch.tensor(destination), 4)
    onehots = torch.cat([taxi_row_onehot, taxi_col_onehot, passenger_onehot, destination_onehot], dim=0)
    return onehots

def distance_pssanger2taxi(taxi_pos: Tuple[int, int], passenger: int):
    passenger_pos_list = {
        0 : (0, 0), 
        1 : (0, 4), 
        2 : (4, 0),
        3 : (4, 3)
    }

    passenger_pos = passenger_pos_list[passenger]

    return ((passenger_pos[0] - taxi_pos[0])**2 + (passenger_pos[1] - taxi_pos[1])**2)**0.5

def conv_reward(state_value: int, reward_value: float) -> float:
    '''
    報酬体系を変更

    乗客との距離が近いほど高い報酬．最大-0.5
    乗客を乗せていると報酬0
    乗客を正しくおろすと+20

    この報酬体系により，なるべく客に近づこうとし，乗客を乗せようとするところまで誘導する

    Args:
        state_value: taxi環境が出力した報酬の値
        reward_value: taxi環境が出力した報酬の値
    '''

    taxi_row, taxi_col, passenger, destination = decode_taxi_state(state_value)

    if passenger == 4:
        return 0.0
    elif reward_value == 20.0:
        return reward_value
    elif reward_value == -10.0:
        return -5.0
    else:
        return -1 + distance_pssanger2taxi((taxi_row, taxi_col), passenger) / 32**0.5


def main():
    env = gym.make("Taxi-v3")

    state_size, action_kinds, action_size, clearScoreThreshold = get_env_info(env)
    print(state_size, action_size, action_kinds, clearScoreThreshold)

    gamma = makeConstant(0.99)
    lr = makeConstant(1e-3)
    tau = makeConstant(1e-2)
    device = torch.device('cpu')
    n_step = 1
    replayBuf = ReplayBuffer(50000, 19, action_size, action_type=torch.int, device=device)
    agent = RainbowAgent(gamma, lr, tau, 
                         19, action_size, action_kinds, 
                         (64, 64, 64), "MSELoss", "Adam", 1, 
                         replayBuf, 128, device, 
                         noisy=True, sigma_init=2.0, 
                         dueling=True, 
                         epsilon_greedy=True)
    
    warmup_Rainbow(env, agent, 200, 10)

    episodes = 50000
    reward_history = list()
    reward_100history = deque(maxlen=100)
    clear_count = 0
    for episode in tqdm(range(episodes), ncols=100):
        done = False
        state, _ = env.reset()
        state_value = state
        state = conv_taxi_state2onehots(state).numpy()
        total_reward = 0.0
        action_history = [0] * action_kinds

        episode_status = list()
        episode_actions = list()
        episode_rewards = list()
        episode_next_status = list()
        episode_dones = list()

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            action = agent.get_action(state_tensor)
            action = action.cpu().detach().numpy().item()

            action_history[action] += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_value = next_state
            next_state = conv_taxi_state2onehots(next_state).numpy()
            # reward = conv_reward(state_value, float(reward))
            reward = np.array(reward, dtype=np.float32)
            done = np.array(truncated or terminated, dtype=np.int8)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            episode_status.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_status.append(next_state)
            episode_dones.append(done)
        
        agent.param_step()
        reward_history.append(total_reward)
        reward_100history.append(total_reward)

        # 良い経験であれば，重複して経験をバッファに追加(成功経験はエピソード長が短いため，バッファを上書きしにくい)
        if len(episode_status) < 0:
            write_count = int(200 / len(episode_status))
            for _ in range(write_count):
                # tqdm.write("重複挿入")
                for state, action, reward, next_state, done in zip(episode_status, episode_actions, episode_rewards, episode_next_status, episode_dones):
                    agent.add_buffer(state, action, reward, next_state, done)

        if total_reward >= clearScoreThreshold:
            clear_count += 1
        else:
            clear_count = 0

        ave_100_reward = sum(reward_100history) / len(reward_100history)
        
        tqdm.write(f"episode: {episode}, reward: {total_reward}")
        tqdm.write(f'episode len: {len(episode_status)}')
        tqdm.write(f'連続クリアカウント: {clear_count}')
        tqdm.write(f'ave rewrad 100 history: {ave_100_reward}')
        tqdm.write(f'action history: {action_history}')
        tqdm.write(f'n step: {n_step}, gamma: {gamma.value}')
        tqdm.write(f'epsilon: {agent._epsilon.value}')
        tqdm.write('')
        
        if ave_100_reward >= clearScoreThreshold:
            print("clear")
            break

    plt.plot(reward_history)
    plt.show()   
    

if __name__ == '__main__':
    main()