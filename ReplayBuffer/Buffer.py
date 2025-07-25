
import numpy as np
from numpy import ndarray
from typing import List, Any, Tuple
import random
from collections import deque

import torch

from usefulParam.Param import ScalarParam
from ReplayBuffer.SamplingTree_pywapper import SamplingTree_pywapper

class ReplayBufferInterface:
    def __init__(self, capacity: int, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        # インスタンス変数
        self._capacity = capacity
        self._device = device

        self._state_size = state_size
        self._action_size = action_size
        self._reward_size = reward_size
        self._done_size = done_size

        self._real_size = 0
        
        # バッファ本体
        self._status = torch.empty(capacity, self._state_size, dtype=state_type, device=self._device)
        self._actions = torch.empty(capacity, self._action_size, dtype=action_type, device=self._device)
        self._rewards = torch.empty(capacity, self._reward_size, dtype=reward_type, device=self._device)
        self._next_status = torch.empty(capacity, self._state_size, dtype=state_type, device=self._device)
        self._dones = torch.zeros(capacity, self._done_size, dtype=done_type, device=self._device)
    
    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        '''
        バッファへ経験を追加する

        Args:
            [state, action, reward, next_state, done]
        '''
        raise NotImplementedError("ReplayBufferInterface.addは仮想関数です")

    
    def get_sample(self, sample_size: int) ->  Any:
        '''
        バッファからサンプリングを行う

        Args:
            size: サンプリングサイズ
        Ret:
            [status, actions, rewards, next_status, dones]
        '''
        raise NotImplementedError("ReplayBufferInterface.get_samplesは仮想関数です")
    
    @property
    def real_size(self):
        return self._real_size
    
    @property
    def capacity(self):
        return self._capacity
    
    def __len__(self):
        return self._real_size
    
    def to(self, device: torch.device):
        '''
            デバイスの変更
        '''
        self._status.to(device)
        self._actions.to(device)
        self._rewards.to(device)
        self._next_status.to(device)
        self._dones.to(device)
    
    def reset(self):
        '''
        初期化
        '''
        raise NotImplementedError("ReplayBufferInterface.resetは仮想関数です")
        

class PERBuffer(ReplayBufferInterface):
    def __init__(self, capacity: int, alpha: ScalarParam, beta: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        super().__init__(capacity, 
                         state_size, action_size, reward_size, done_size, 
                         state_type, action_type, reward_type, done_type, 
                         device)
        self._alpha = alpha
        self._beta = beta

        self._priorities = SamplingTree_pywapper(self._capacity)

        self._indics4update = np.empty(0)
    
    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        # 優先度更新
        write_idx = 0
        if self._real_size == 0:
            write_idx = self._priorities.add(1.0)
        else:
            write_idx = self._priorities.add(self._priorities.max_leaf)
        
        # バッファ更新
        self._status[write_idx] = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        self._actions [write_idx] = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        self._rewards[write_idx] = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        self._next_status[write_idx] = torch.tensor(next_state, dtype=self._next_status.dtype, device=self._device)
        self._dones[write_idx] = torch.tensor(done, dtype=self._dones.dtype, device=self._device)

        self._real_size = min(self._real_size + 1, self._capacity)

    def get_sample(self, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        priorities, indics = self._priorities.get_sample(sample_size)
        weights = self._calc_weight(priorities)

        self._indics4update = np.array(indics)

        extracted_status = self._status[indics]
        extracted_actions = self._actions[indics]
        extracted_rewards = self._rewards[indics]
        extracted_next_status = self._next_status[indics]
        extracted_dones = self._dones[indics]

        return extracted_status, extracted_actions, extracted_rewards, extracted_next_status, extracted_dones, weights
    
    def _calc_weight(self, priorities: ndarray) -> torch.Tensor:
        '''
        優先度から重みを計算する
        '''
        # 優先度全体の合計を取得する(c++へのアクセスを1度に抑えるために記録)
        priority_total = self._priorities.total

        # 経験の選択確率を計算する
        select_probs = priorities / priority_total

        # 重みの計算
        weights = (self.real_size * select_probs)**self._beta.value

        # 正規化
        weights /= np.max(weights)

        return torch.tensor(weights, device=self._device).unsqueeze(1)

    def update(self, td_diffs: torch.Tensor) -> None:
        '''
        優先度を更新する
        '''
        if self._indics4update.shape == (0,):
            raise RuntimeError("まだ，サンプリングが行われていません")

        # 優先度の計算
        td_diffs = td_diffs.squeeze(1)
        new_priorities = self._calc_priorities(td_diffs.detach().cpu().numpy())

        # 更新
        self._priorities.update(new_priorities, self._indics4update)
        self._indics4update = np.empty(0)

    def _calc_priorities(self, td_diffs: ndarray) -> ndarray:
        '''
        TD誤差から優先度を計算する
        '''
        priorities = (td_diffs + 1e-6)**self._alpha.value
        return priorities
    
    def step_param(self):
        '''
        パラメータのステップ
        '''
        self._alpha.step()
        self._beta.step()
    
    def reset(self):
        self = None

class ReplayBuffer(ReplayBufferInterface):
    '''
    リプレイバッファ
    '''
    def __init__(self, capacity: int, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        super().__init__(capacity, 
                         state_size, action_size, reward_size, done_size, 
                         state_type, action_type, reward_type, done_type, 
                         device)
        self._write_idx = 0

    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        '''
        バッファへ要素を追加する

        Args:
            observation: バッファへ加える要素
            ovservation = [state, action, reward, next_state, done]
        '''

        self._status[self._write_idx] = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        self._actions[self._write_idx] = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        self._rewards[self._write_idx] = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        self._next_status[self._write_idx] = torch.tensor(next_state, dtype=self._next_status.dtype, device=self._device)
        self._dones[self._write_idx] = torch.tensor(done, dtype=self._dones.dtype, device=self._device)
        
        self._write_idx = (self._write_idx + 1) % self._capacity
        self._real_size = min(self._real_size + 1, self._capacity)

    def get_sample(self, sample_size: int) -> list[torch.Tensor]:
        '''
        バッチの取り出し
        Args:
            batch_size: 取り出すバッチサイズ
        Ret:
            取り出したバッチ(Batch)
        '''
        indics = random.sample(range(self._real_size), sample_size)

        extract_status = self._status[indics]
        extract_actions = self._actions[indics]
        extract_rewards = self._rewards[indics]
        extract_next_status = self._next_status[indics]
        extract_dones = self._dones[indics]

        batch = [extract_status, extract_actions, extract_rewards, extract_next_status, extract_dones]

        return batch
    
    def to(self, device: torch.device):
        '''
            デバイスの変更
        '''
        self._status.to(device)
        self._actions.to(device)
        self._rewards.to(device)
        self._next_status.to(device)
        self._dones.to(device)

################################################## N step リプレイバッファ ##################################################

class NstepSampleMaker:
    '''
    N stepのSampleを作成する
    '''
    def __init__(self, n_step: int, gamma: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device('cpu')):
        self._device = device

        self._n_step = n_step
        self._gamma = gamma

        self._state_size = state_size
        self._action_size = action_size
        self._reward_size = reward_size
        self._done_size = done_size

        self._state_type = state_type
        self._action_type = action_type
        self._reward_type = reward_type
        self._done_type = done_type

        self._status = deque(maxlen=self._n_step)
        self._actions = deque(maxlen=self._n_step)
        self._rewards = deque(maxlen=self._n_step)
        self._next_state = torch.empty(0, device=self._device)
        self._done = torch.tensor(0, dtype=self._done_type, device=self._device)
    
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        '''
        経験を追加する

        Args: 
            state, action, reward, next_state, done:
                入力する経験．deviceは変えないため，事前に適切なデバイスに入れておく必要がある
        '''
        if self._done:
            self._status.clear()
            self._actions.clear()
            self._rewards.clear()
            self._next_state = torch.empty(0, device=self._device)
            self._done = torch.tensor(0, dtype=torch.int8, device=self._device)

        # 経験の連続性を確認
        if not torch.equal(self._next_state, state) and len(self._status) != 0:
            raise RuntimeError("経験が連続的ではありません")

        self._status.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_state = next_state
        self._done = done
    
    def get_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        サンプルを作成する
        '''
        if not self.can_makeSample():
            raise RuntimeError("経験がサンプルのサイズ分たまっていません")

        # サンプル構成
        cnt_gamma = 1.0
        sample_state = torch.tensor(self._status[0], dtype=self._state_type, device=self._device)
        sample_action = torch.tensor(self._actions[0], dtype=self._action_type, device=self._device)
        sample_reward = torch.zeros(self._reward_size, dtype=self._reward_type, device=self._device)
        sample_next_state = torch.tensor(self._next_state, dtype=self._state_type, device=self._device)
        sample_done = torch.tensor(self._done, dtype=self._done_type, device=self._device)

        for reward in self._rewards:
            sample_reward += cnt_gamma * reward
            cnt_gamma *= self._gamma.tensor_value

        return sample_state, sample_action, sample_reward, sample_next_state, sample_done
        

    def can_makeSample(self):
        '''
        サンプルが作成可能か．
        すなわち，N個の経験がたまっているか
        '''
        return len(self._status) == self._n_step

class BaseNstepReplayBuffer:
    '''
    N stepリプレイバッファの共通部分
    '''
    def __init__(self, capacity: int, n_step: int, gamma: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device('cpu')):
        self._device = device

        self._capacity = capacity
        self._n_step = n_step
        self._gamma = gamma

        self._state_size = state_size
        self._action_size = action_size
        self._reward_size = reward_size
        self._done_size = done_size

        self._real_size = 0

        self._status = torch.empty(self._capacity, self._state_size, dtype=state_type, device=self._device)
        self._actions = torch.empty(self._capacity, self._action_size, dtype=action_type, device=self._device)
        self._rewards = torch.empty(self._capacity, self._reward_size, dtype=reward_type, device=self._device)
        self._next_status = torch.empty(self._capacity, self._state_size, dtype=state_type, device=self._device)
        self._dones = torch.empty(self._capacity, self._done_size, dtype=done_type, device=self._device)

        self._sampleMaker = NstepSampleMaker(self._n_step, self._gamma, self._state_size, self._action_size, self._reward_size, self._done_size, 
                                       state_type, action_type, reward_type, done_type, 
                                       self._device)
    
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def real_size(self):
        return self._real_size
    
    @property
    def n_step(self):
        return self._n_step
    
    def __len__(self):
        return self._real_size
    
    def to(self, device: torch.device):
        '''
        デバイスの移動
        '''
        self._status = self._status.to(device)
        self._actions = self._actions.to(device)
        self._rewards = self._rewards.to(device)
        self._dones = self._dones.to(device)
        self._gamma.to(device)
        

class NstepPERBuffer(BaseNstepReplayBuffer):
    '''
    N step PER リプレイバッファ
    '''
    def __init__(self, capacity: int, n_step: int, gamma: ScalarParam, 
                 alpha: ScalarParam, beta: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device('cpu')):
        super().__init__(capacity, n_step, gamma, 
                         state_size, action_size, reward_size, done_size, 
                         state_type, action_type, reward_type, done_type, device)
        
        self._alpha = alpha
        self._beta = beta

        self._priorities = SamplingTree_pywapper(self._capacity)

        self._indics4update = np.empty(0)
    
    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        '''
        バッファへ経験を追加する
        '''        
        
        # tensorへ変換
        state_tensor = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        action_tensor = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        reward_tensor = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        next_state_tensor = torch.tensor(next_state, dtype=self._next_status.dtype, device=self._device)
        done_tensor = torch.tensor(done, dtype=self._dones.dtype, device=self._device)

        # n step makerへ経験を格納
        self._sampleMaker.add(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)

        # makerにN stepたまっていれば，バッファへ追加
        if (not self._sampleMaker.can_makeSample()):
            return 
        
        self._write_to_buffer(*self._sampleMaker.get_sample())
    
    def _write_to_buffer(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor):
        '''
        バッファへ書き込む
        '''

        write_idx = 0
        if self._real_size == 0:
            write_idx = self._priorities.add(1.0)
        else:
            write_idx = self._priorities.add(self._priorities.max_leaf)

        self._status[write_idx] = state
        self._actions [write_idx] = action
        self._rewards[write_idx] = reward
        self._next_status[write_idx] = next_state
        self._dones[write_idx] = done

        self._real_size = min(self._real_size + 1, self._capacity)

    def get_sample(self, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        priorities, indics = self._priorities.get_sample(sample_size)
        weights = self._calc_weight(priorities)

        self._indics4update = np.array(indics)

        extracted_status = self._status[indics]
        extracted_actions = self._actions[indics]
        extracted_rewards = self._rewards[indics]
        extracted_next_status = self._next_status[indics]
        extracted_dones = self._dones[indics]

        return extracted_status, extracted_actions, extracted_rewards, extracted_next_status, extracted_dones, weights
    
    def _calc_weight(self, priorities: ndarray) -> torch.Tensor:
        '''
        優先度から重みを計算する
        '''
        # 優先度全体の合計を取得する(c++へのアクセスを1度に抑えるために記録)
        priority_total = self._priorities.total

        # 経験の選択確率を計算する
        select_probs = priorities / priority_total

        # 重みの計算
        weights = (self.real_size * select_probs)**self._beta.value

        # 正規化
        weights /= np.max(weights)

        return torch.tensor(weights, device=self._device).unsqueeze(1)

    def update(self, td_diffs: torch.Tensor) -> None:
        '''
        優先度を更新する
        '''
        if self._indics4update.shape == (0,):
            raise RuntimeError("まだ，サンプリングが行われていません")

        # 優先度の計算
        td_diffs = td_diffs.squeeze(1)
        new_priorities = self._calc_priorities(td_diffs.detach().cpu().numpy())

        # 更新
        self._priorities.update(new_priorities, self._indics4update)
        self._indics4update = np.empty(0)

    def _calc_priorities(self, td_diffs: ndarray) -> ndarray:
        '''
        TD誤差から優先度を計算する
        '''
        priorities = (td_diffs + 1e-6)**self._alpha.value
        return priorities
    
    def step_param(self):
        '''
        パラメータのステップ
        '''
        self._alpha.step()
        self._beta.step()

    def to(self, device: torch.device):
        super().to(device)
        self._alpha.to(device)
        self._beta.to(device)

class NstepReplayBuffer(BaseNstepReplayBuffer):
    '''
    N step リプレイバッファ
    '''
    def __init__(self, capacity: int, n_step: int, gamma: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device('cpu')):
        super().__init__(capacity, n_step, gamma, state_size,action_size, reward_size, done_size, 
                         state_type, action_type, reward_type, done_type, device)
        self._write_idx = 0
    
    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        '''
        経験の追加
        '''
        state_tensor = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        action_tensor = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        reward_tensor = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        next_state_tensor = torch.tensor(next_state, dtype=self._status.dtype, device=self._device)
        done_tensor = torch.tensor(done, dtype=self._dones.dtype, device=self._device)

        self._sampleMaker.add(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)

        if (not self._sampleMaker.can_makeSample()):
            return
                
        self.write2buffer(*self._sampleMaker.get_sample())

    def write2buffer(self, status: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        self._status[self._write_idx] = status
        self._actions[self._write_idx] = actions
        self._rewards[self._write_idx] = reward
        self._next_status[self._write_idx] = next_state
        self._dones[self._write_idx] = done

        self._write_idx = (self._write_idx + 1) % self._capacity
        self._real_size = min(self._real_size + 1, self._capacity)
    
    def get_sample(self, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        サンプルの抜き出し
        '''
        indices = np.random.choice(range(self._real_size), sample_size)

        extract_status = self._status[indices]
        extract_actions = self._actions[indices]
        extract_rewards = self._rewards[indices]
        extract_next_status = self._next_status[indices]
        extract_dones = self._dones[indices]

        return extract_status, extract_actions, extract_rewards, extract_next_status, extract_dones

############################## テスト##############################

from usefulParam.Param import ScalarParam, makeConstant
def test_NstepReplayBuffer():
    buf = NstepReplayBuffer(2000, 3, makeConstant(0.99), 4, 1)

    episodes = 20
    for episode in range(episodes):
        state_count = 0
        for i in range(200):
            state = np.array([state_count] * 4)
            action = np.random.choice(range(4))
            reward = np.array(state_count)
            next_state = np.array([state_count + 1] * 4)
            done = np.array(True if i == 199 else False)
            buf.add(state, action, reward, next_state, done)

            state_count += 1

            if buf.real_size >= 2:
                sample = buf.get_sample(2)
                status, actions, rewards, next_status, dones = sample
                print(f'state: {status}')
                print(f'action: {actions}')
                print(f'reward: {rewards}')
                print(F'next state: {next_status}')
                print(f'done: {dones}')
    