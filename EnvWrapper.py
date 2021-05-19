from rlpyt.envs.gym import *
import gym 

class RewarableEnv () : 

    def __init__ (self, env, reward=None, internalStateFn=lambda x : x, s0=None) :
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        if reward is None : 
            self.reward_range = env.reward_range
        else :
            self.reward_range = reward.reward_range
        self.reward = reward
        self.internalStateFn = internalStateFn
        self.s0 = s0

    def step (self, a) : 
        s, r, d, i = self.env.step(a)
        if self.reward is None: 
            return s, r, d, i
        else :
            return s, self.reward(s), d, i

    def reset (self) : 
        o = self.env.reset()
        if self.s0 is not None :
            self.env.env.state = self.internalStateFn(self.s0)
            o = self.s0
        return o

    def render (self, mode='Human') : 
        return self.env.render(mode)

    def close (self) :
        self.env.close()

    def seed (self, seed=None) : 
        self.env.seed(seed)

    def setRewardFn(self, reward) : 
        self.reward = reward

    def setState (self, state) : 
        self.env.state = self.internalStateFn(state)

def rlpyt_make(*args, info_example=None, reward=None, s0=None,
               internalStateFn=lambda x: x, **kwargs):
    env  = gym.make(*args, **kwargs)
    renv = RewarableEnv(env, reward, internalStateFn, s0)
    if info_example is None:
        return GymEnvWrapper(renv)
    else:
        return GymEnvWrapper(EnvInfoWrapper(renv, info_example))
