from gymnasium import Env, spaces
import numpy as np
from typing import Tuple, Optional, Dict

def get_env_info(env: Env) -> Tuple[int, int, int, float]:
    """
    Env インスタンスから次の情報を取得して dict で返す：
      - state_dim: flatten した観測ベクトル長
      - action_dim: 離散アクションの種類数
      - action_input_dim: 環境に渡すアクションの次元 (離散ならスカラー 1)
      - reward_threshold: クリアの報酬閾値 (未定義なら None)

    Args:
        env (Env): gymnasium.Env のインスタンス

    Returns:
        info (dict):
          {
            "state_dim": int,
            "action_dim": int,
            "action_input_dim": int,
            "reward_threshold": Optional[float]
          }
    """
    obs_space = env.observation_space
    act_space = env.action_space

    # --- state_dim ---
    if isinstance(obs_space, spaces.Box):
        state_dim = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, spaces.Discrete):
        state_dim = obs_space.n
    else:
        raise ValueError(f"Unsupported observation space: {obs_space}")

    # --- action_dim / action_input_dim ---
    if isinstance(act_space, spaces.Discrete):
        action_dim = act_space.n
        action_input_dim = 1
    else:
        raise ValueError(f"Unsupported action space: {act_space}")

    # --- reward_threshold ---
    # クリア基準は env.spec.reward_threshold に定義されている場合がある
    reward_threshold = None
    if env.spec is not None:
        reward_threshold = env.spec.reward_threshold

    return int(state_dim), int(action_dim), int(action_input_dim), reward_threshold

if __name__ == "__main__":
    import gymnasium as gym

    for name in ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "FrozenLake-v1", "LunarLander-v3"]:
        env = gym.make(name)
        info = get_env_info(env)
        print(f"{name:15} →", info)
        env.close()