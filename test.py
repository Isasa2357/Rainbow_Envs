
import gymnasium as gym

import torch
from usefulParam.Param import makeConstant, makeMultiply

# from DQN.NoisyNet import test_NoisyNet, test_DuelingNoisyNet, test_commDefine
# from ReplayBuffer.Buffer import test_NstepReplayBuffer
from ReplayBuffer.Buffer_v2 import test_NstepReplayBuffer, NstepReplayBuffer
from DQN.Rainbow import warmup_Rainbow, RainbowAgent


# test_NoisyNet()
# test_DuelingNoisyNet()
# test_commDefine()

env = gym.make("LunarLander-v3")
gamma = 0.99
n_step = 3
state_size = 8
action_size = 4
action_kinds = 1
device =torch.device('cpu')
# replayBuf = ReplayBuffer(20000, state_size, action_size, action_type=torch.int, device=device)
replayBuf = NstepReplayBuffer(20000, n_step, makeConstant(gamma, device), state_size, action_size, action_type=torch.int, device=device)
agent = RainbowAgent(makeConstant(gamma, device), makeConstant(1e-4, device), makeConstant(5e-3, device), 
                        state_size, action_size, action_kinds, 
                        (64, 64, 64), "MSELoss", "Adam", 1, 
                        replayBuf, 64, device, 
                        noisy=True, sigma_init=0.3, epsilon=makeMultiply(1.0, 0.995, 1e-4, 1.0, device), 
                        dueling=True)

if __name__ == '__main__':
    import time
    start = time.time()
    warmup_Rainbow(env, agent, 1000, 10)
    end = time.time()

    print(end- start)
    print(agent._replayBuf.real_size)