# IRL

The Inverse Reinforcement Learning problem asks why some observed behaviour happens. Let's say, I'm playing football and I run down the pitch and kick the ball towards the opponent's net. With access to only such behaviours, with no other knowledge of the world, the IRL problem determines the reward structure that is incentivizing this behaviour. In football, there is the reward of scoring a goal. Like in many other cases, this reward is sparse (there are few goals per 90 minutes) and serves as a succint representation of the task. Here I'm building a tool for testing different solutions on a variety of environments. 

Currently, I have one solution for one environment. The solution is an [LP formulation](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) for finding an optimal reward function from a linearly parameterized family of reward functions. The environment is the [Acrobot-v1](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py) from OpenAI Gym.
 
## Setup

I have used [rlpyt](https://github.com/astooke/rlpyt) for finding optimal policies for different reward functions. It is instantiated as a submodule. So you'll have to recursively clone this repository and then build rlpyt.

```
$ git clone --recurse https://github.com/Vrroom/IRL.git
$ cd IRL/rlpyt
$ python3 setup.py install
$ cd .. && python3 IRL.py
```

## Example

The Acrobot-v1 is a double pendulum system. An agent can give clockwise or counterclockwise torque. The goal is to get the bottom link at a particular height. I played with this environment and accumulated 100 trajectories. These were the inputs to the IRL algorithm.

The reward function is specified as a function of the angle that top link makes with the vertical `theta1` and the angle the bottom link makes with respect to the top link `theta2`. `theta1` and `theta2` are in the range `[-PI, PI]`. The plot on the left in the table below is the true reward function. The plot on the right is the reward function recovered from observed behaviour. Even though the two plots don't match over the whole domain, they match in a crucial region of the domain. The points near the origin correspond to the starting position, where `theta1 = theta2 = 0` and both the links point downwards. Both reward functions incentivize getting out of this region. My guess is that once the agent is out of this region, it has acquired sufficient kinetic energy such that it will finish the task eventually, even if no further action is taken.

<table>
  <tr>
    <td> <img width="256" src="https://user-images.githubusercontent.com/7254326/118997340-36cc4b00-b9a6-11eb-9477-572415c4b647.png" /> </td>
    <td> <img width="256" src="https://user-images.githubusercontent.com/7254326/119001218-6a5ca480-b9a9-11eb-9e90-022e07d77748.png" /> </td>
  </tr>
</table>

The optimal policy under the recovered reward function can be seen below. This policy is also accomplishes the task. 

![acrobot](https://user-images.githubusercontent.com/7254326/119007009-80209880-b9ae-11eb-811d-e6ba0f3d169c.gif)

## Algorithm Overview 

We assume that the observed policy is optimal for the underlying task and iteratively shrink the space of reward function candidates. Initially, we guess a reward function and find the optimal policy for it. Valid candidates for reward function satisfy that the value of the states under this new policy is less than or equal to that under the observed policy. This constraint shrinks the space of valid candidates. We find the *best* valid reward candidate, characterized by superiority of value of observed policy over the new policy. An optimal policy is determined for this new candidate and the cycle is repeated again. The outer loop described above can be found [here](https://github.com/Vrroom/IRL/blob/cd5f112ad4728fd12a19c09a7670ef037f6a00bc/IRL.py#L101). Reward Space shrinking can be found [here](https://github.com/Vrroom/IRL/blob/cd5f112ad4728fd12a19c09a7670ef037f6a00bc/RewardFnSpace.py#L102).
 
## Future Work

1. I think that the main bottleneck stopping this work from being generalized to other environments is the reward bases are tied to the state space in these environments. There is no general way of choosing reward bases for all environments.
2. Add Ziebart's _Maximum Entropy Inverse Reinforcement Learning_ and the more recent _Maximum Entropy Deep Inverse Reinforcement Learning_ by Wulfmeier.
3. One criticism I have of this field of research is that it doesn't actually solve its motivation. The motivation was to recover the succint representation of the task from observed behavior. For example, "score a goal" or "cross the grey line". As we saw in the example, the reward function that we recovered, although explained the observed behaviour, didn't convey the meaning of the original reward function. This begs the question, what additional knowledge about the environment, does solving this problem give?

## References

1. [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
2. [Linear Programming for Piecewise Linear Objectives](http://www.seas.ucla.edu/~vandenbe/ee236a/lectures/pwl.pdf)
