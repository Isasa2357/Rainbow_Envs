
from typing import Tuple, Dict, List

import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import deque
import json

import torch
from torch.nn import functional as F

import gymnasium as gym
from gymnasium import Env


from mutil_RL.mutil_gym import get_env_info
from ReplayBuffer.Buffer_v2 import ReplayBuffer, NstepReplayBuffer, PERBuffer, NstepPERBuffer
from DQN.Rainbow import RainbowAgent, warmup_Rainbow
from usefulParam.Param import makeConstant, makeMultiply
from mutil_common.mutil_common import resolve_conflict_filename

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

def makeResult(args: Dict, reward_history: List, q_loss_history: List):
    taxi_resultFolder = 'Taxi'
    project_path = os.path.join(taxi_resultFolder, args['project'])
    resultFolder = resolve_conflict_filename(args['result'], project_path)
    resultFolder_path = os.path.join(project_path, resultFolder)

    os.makedirs(taxi_resultFolder, exist_ok=True)
    os.makedirs(project_path, exist_ok=True)
    os.mkdir(resultFolder_path)

    # 条件を記録
    with open(os.path.join(resultFolder_path, 'condition.json'), 'w') as cj:
        json.dump(args, cj, indent=4)
    
    # 報酬の推移を記録
    with open(os.path.join(resultFolder_path, 'reward_history.txt'), 'w') as f:
        for reward in reward_history:
            f.write(str(reward))
            f.write('\n')
    plt.plot(reward_history)
    plt.savefig(os.path.join(resultFolder_path, 'reward_history.png'))
    plt.clf()

    # 損失の推移を記録
    with open(os.path.join(resultFolder_path, 'qLoss_history.txt'), 'w') as qlhf:
        for loss in q_loss_history:
            qlhf.write(str(loss))
            qlhf.write('\n')
    plt.plot(q_loss_history)
    plt.savefig(os.path.join(resultFolder_path, 'qLoss_history.png'))
    plt.clf()


def main():
    args = {
        'episodes': 10000, 
        'wormup_episodes': 200, 
        'wormup_workers': 10, 
        'gamma': 0.99, 
        'lr': 1e-3, 
        'tau': 5e-3, 
        'n_step': 1, 
        'hdn_lays': (64, 64, 64), 
        'lossF': 'MSELoss', 
        'optimizer': 'Adam', 
        'sync_interval': 1, 
        'buf_capacity': 30000, 
        'batch_size': 128, 
        'device': 'cpu', 
        'noisy': True, 
        'sigma_init': 2.0, 
        'dueling': True, 
        'epsilon_greedy': True, 
        'epsilon': (1.0, 0.998, 1e-4, 1.0), 
        'project': 'epsilonInit0.4', 
        'result': 'result'
    }

    env = gym.make("Taxi-v3")

    state_size, action_kinds, action_size, clearScoreThreshold = get_env_info(env)
    print(state_size, action_size, action_kinds, clearScoreThreshold)

    gamma = makeConstant(args['gamma'])
    lr = makeConstant(args['lr'])
    tau = makeConstant(args['tau'])
    device = torch.device(args['device'])
    n_step = args['n_step']
    replayBuf = ReplayBuffer(args['buf_capacity'], 19, action_size, action_type=torch.int, device=device)
    agent = RainbowAgent(gamma, lr, tau, 
                         19, action_size, action_kinds, 
                         args['hdn_lays'], args['lossF'], args['optimizer'], args['sync_interval'], 
                         replayBuf, args['batch_size'], device, 
                         noisy=args['noisy'], sigma_init=args['sigma_init'], 
                         dueling=args['dueling'], 
                         epsilon_greedy=args['epsilon_greedy'], epsilon=makeMultiply(*args['epsilon']))
    
    warmup_Rainbow(env, agent, args['wormup_episodes'], args['wormup_workers'])

    episodes = args['episodes']
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
        reward_history.append(float(total_reward))
        reward_100history.append(total_reward)

        ave_100_reward = sum(reward_100history) / len(reward_100history)
        
        tqdm.write(f"episode: {episode}, reward: {total_reward}")
        tqdm.write(f'episode len: {len(episode_status)}')
        tqdm.write(f'ave rewrad 100 history: {ave_100_reward}')
        tqdm.write(f'action history: {action_history}')
        tqdm.write(f'n step: {n_step}, gamma: {gamma.value}')
        tqdm.write(f'epsilon: {agent._epsilon.value}')
        tqdm.write('')
        
        if ave_100_reward >= clearScoreThreshold:
            print("clear")
            break

    makeResult(args, reward_history, agent.q_net_loss_history)


    

if __name__ == '__main__':
    main()