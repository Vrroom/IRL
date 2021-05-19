class Reward () :
    def __init__ (self, rewardFn, reward_range): 
        self.rewardFn = rewardFn
        self.reward_range = reward_range

    def __call__ (self, s) :
        r = self.rewardFn(s)
        lo, hi = self.reward_range
        assert(lo <= r <= hi)
        return r


