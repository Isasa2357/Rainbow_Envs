import gymnasium as gym



def main():
    env = gym.make("ALE/Breakout-v5")

    state_size = (84, 84, 4)
    action_kinds = 4
    action_size = 1