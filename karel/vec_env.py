import gym
from gym import spaces

class VecEnv(gym.Env):
    def __init__(self, action_space = None, observation_space = None) -> None:
        super(VecEnv, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space