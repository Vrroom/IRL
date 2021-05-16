from rlpyt.envs.gym import *
import gym 

class RewarableEnv () : 

    def __init__ (self, env, reward=None) :
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        if reward is None : 
            self.reward_range = env.reward_range
        else :
            self.reward_range = reward.reward_range
        self.reward = reward

    def step (self, a) : 
        s, r, d, i = self.env.step(a)
        if self.reward is None: 
            return s, r, d, i
        else :
            return s, self.reward(s), d, i

    def reset (self) : 
        return self.env.reset()

    def render (self, mode='Human') : 
        return self.env.render(mode)

    def close (self) :
        self.env.close()

    def seed (self, seed=None) : 
        self.env.seed(seed)

    def setRewardFn(self, reward) : 
        self.reward = reward

def rlpyt_make(*args, info_example=None, reward=None, **kwargs):
    env  = gym.make(*args, **kwargs)
    renv = RewarableEnv(env, reward)
    if info_example is None:
        return GymEnvWrapper(renv)
    else:
        return GymEnvWrapper(EnvInfoWrapper(renv, info_example))
