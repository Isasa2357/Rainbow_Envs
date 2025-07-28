
import numpy as np
from numpy import ndarray
from typing import List, Any, Tuple
import random
from collections import deque

import torch

from usefulParam.Param import ScalarParam
from ReplayBuffer.SamplingTree_pywapper import SamplingTree_pywapper

############################## N step maker ##############################

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
        if self._done == 1:
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
        # sample_state = torch.tensor(self._status[0], dtype=self._state_type, device=self._device)
        # sample_action = torch.tensor(self._actions[0], dtype=self._action_type, device=self._device)
        # sample_reward = torch.zeros(self._reward_size, dtype=self._reward_type, device=self._device)
        # sample_next_state = torch.tensor(self._next_state, dtype=self._state_type, device=self._device)
        # sample_done = torch.tensor(self._done, dtype=self._done_type, device=self._device)

        sample_state = self._status[0]
        sample_action = self._actions[0]
        sample_reward = torch.zeros(self._reward_size, dtype=self._reward_type, device=self._device)
        sample_next_state = self._next_state
        sample_done = self._done

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

############################## ReplayBuffer ##############################

class BaseReplayBuffer:
    '''
    ReplayBufferとPERの共通属性
    '''
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
    
    def write2buffer(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor, write_idx:int) -> None:
        if (state == None) or (action == None) or (reward == None) or (next_state == None) or (done == None):
            raise RuntimeError("observationにNoneが含まれる")

        self._status[write_idx] = state
        self._actions[write_idx] = action
        self._rewards[write_idx] = reward
        self._next_status[write_idx] = next_state
        self._dones[write_idx] = done

        self._step_real_size()

    def get_sample(self, sample_size: int) ->  Any:
        '''
        バッファからサンプリングを行う

        Args:
            size: サンプリングサイズ
        Ret:
            [status, actions, rewards, next_status, dones]
        '''
        raise NotImplementedError("ReplayBufferInterface.get_samplesは仮想関数です")
    
    def _step_real_size(self) -> None:
        self._real_size = min(self._real_size + 1, self._capacity)
    
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
        self._status = self._status.to(device)
        self._actions = self._actions.to(device)
        self._rewards = self._rewards.to(device)
        self._next_status = self._next_status.to(device)
        self._dones = self._dones.to(device)

class ReplayBuffer(BaseReplayBuffer):
    '''
    ランダムサンプリングを行う，通常のリプレイバッファの共通属性
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
        # 経験の各要素をTensorに変換
        state_tensor = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        action_tensor = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        reward_tensor = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        next_state_tensor = torch.tensor(next_state, dtype=self._status.dtype, device=self._device)
        done_tensor = torch.tensor(done, dtype=self._dones.dtype, device=self._device)

        # 経験を変換(拡張機能)
        state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, do_write = self._conv_observation_4add(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)

        if not do_write:
            return

        self.write2buffer(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, self._write_idx)
    
    def _conv_observation_4add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, bool]:
        '''
        経験をバッファに追加する前に変更する
        拡張機能のため
        例:
            N stepの場合，ここでN stepを管理・出力する

        Ret:
            convd_state, convd_action, convd_reward, convd_next_state, convd_done, do_write
            do_write: 書き込みを行うか
        '''
        return state, action, reward, next_state, done, True
    
    def write2buffer(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor, write_idx: int) -> None:
        '''
        バッファへ経験を書き込む
        '''
        super().write2buffer(state, action, reward, next_state, done, write_idx)
        self._step_write_idx()
    
    def get_sample(self, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        サンプルの取り出し
        Args:
            sample_size: 取り出すバッチサイズ
        Ret:
            samples[batch, each_size]
            status, actions, rewards, next_status, done
        '''
        indics = self._get_indices(sample_size)

        extract_status = self._status[indics]
        extract_actions = self._actions[indics]
        extract_rewards = self._rewards[indics]
        extract_next_status = self._next_status[indics]
        extract_dones = self._dones[indics]

        return extract_status, extract_actions, extract_rewards, extract_next_status, extract_dones
    
    def _get_indices(self, sample_size: int):
        '''
        抽出するサンプルの位置を決める
        '''
        return random.sample(range(self._real_size), sample_size)

    def _step_write_idx(self) -> None:
        self._write_idx = (self._write_idx + 1) % self._capacity

class NstepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, n_step: int, gamma: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        super().__init__(capacity, 
                         state_size, action_size, reward_size, done_size,
                         state_type, action_type, reward_type, done_type, 
                         device)
        
        self._n_step = n_step
        self._maker = NstepSampleMaker(self._n_step, gamma, 
                                       state_size, action_size, reward_size, done_size, 
                                       state_type, action_type, reward_type, done_type, device)
    
    def _conv_observation_4add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, bool]:
        self._maker.add(state, action, reward, next_state, done)

        if self._maker.can_makeSample():
            return *self._maker.get_sample(), True
        else:
            return None, None, None, None, None, False
    
    @property
    def n_step(self):
        return self._n_step
    
class PERBuffer(BaseReplayBuffer):
    def __init__(self, capacity: int, alpha: ScalarParam, beta: ScalarParam, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        super().__init__(capacity, 
                         state_size, action_size, reward_size, done_size, 
                         state_type, action_type, reward_type, done_type, device)
        self._alpha = alpha
        self._beta = beta

        self._priorities = SamplingTree_pywapper(self._capacity)

        self._indices4update = np.empty(0)
    
    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        
        # 経験の各要素をTensorに変換
        state_tensor = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        action_tensor = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        reward_tensor = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        next_state_tensor = torch.tensor(next_state, dtype=self._status.dtype, device=self._device)
        done_tensor = torch.tensor(done, dtype=self._dones.dtype, device=self._device)

        state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, do_write = self._conv_observation_4add(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)

        if not do_write:
            return

        # 優先度更新
        write_idx = 0
        if self._real_size == 0:
            write_idx = self._priorities.add(1.0)
        else:
            write_idx = self._priorities.add(self._priorities.max_leaf)
        
        # バッファ更新
        super().write2buffer(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, write_idx)
    
    def _conv_observation_4add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, bool]:
        '''
        経験をバッファに追加する前に変更する
        拡張機能のため
        例:
            N stepの場合，ここでN stepを管理・出力する

        Ret:
            convd_state, convd_action, convd_reward, convd_next_state, convd_done, do_write
            do_write: 書き込みを行うか
        '''
        return state, action, reward, next_state, done, True
    
    def get_sample(self, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        priorities, indics = self._priorities.get_sample(sample_size)
        weights = self._calc_weight(priorities)

        self._indices4update = np.array(indics)

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
        if self._indices4update.shape == (0,):
            raise RuntimeError("まだ，サンプリングが行われていません")

        # 優先度の計算
        td_diffs = td_diffs.squeeze(1)
        new_priorities = self._calc_priorities(td_diffs.detach().cpu().numpy())

        # 更新
        self._priorities.update(new_priorities, self._indices4update)
        self._indices4update = np.empty(0)
    
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

class NstepPERBuffer(PERBuffer):
    def __init__(self, capacity: int, alpha: ScalarParam, beta: ScalarParam, gamma: ScalarParam, n_step: int, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        super().__init__(capacity, alpha, beta, 
                         state_size, action_size, reward_size, done_size, 
                         state_type, action_type, reward_type, done_type, device)
        
        self._gamma = gamma 
        self._n_step = n_step
        self._maker = NstepSampleMaker(self._n_step, self._gamma, 
                                       state_size, action_size, reward_size, done_size, 
                                       state_type, action_type, reward_type, done_type, device)
        
    def _conv_observation_4add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, bool]:
        self._maker.add(state, action, reward, next_state, done)

        if self._maker.can_makeSample():
            return *self._maker.get_sample(), True
        else:
            return None, None, None, None, None, False

############################## テスト##############################

from usefulParam.Param import ScalarParam, makeConstant
def test_NstepReplayBuffer():
    # buf = NstepReplayBuffer(2000, 3, makeConstant(0.99), 4, 1)
    # buf = NstepPERBuffer(2000, makeConstant(0.4), makeConstant(1.0), makeConstant(0.99), 3, 4, 1)
    # buf = ReplayBuffer(2000, 4, 1)
    buf = PERBuffer(2000, makeConstant(0.4), makeConstant(1.0), 4, 1)

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
                if isinstance(buf, ReplayBuffer) or isinstance(buf, NstepReplayBuffer):
                    sample = buf.get_sample(2)
                    status, actions, rewards, next_status, dones = sample
                elif isinstance(buf, PERBuffer) or isinstance(buf, NstepPERBuffer):
                    sample = buf.get_sample(2)
                    status, actions, rewards, next_status, dones, weight = sample

                print(f'state: {status}')
                print(f'action: {actions}')
                print(f'reward: {rewards}')
                print(F'next state: {next_status}')
                print(f'done: {dones}')
    