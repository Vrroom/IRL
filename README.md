# IRL

The Inverse Reinforcement Learning problem asks why some observed behaviour happens. Let's say, I'm playing football and I run down the pitch and kick the ball towards the opponent's net. With access to only such behaviours, with no other knowledge of the world, the IRL problem determines the reward structure that is incentivizing this behaviour. In football, the reward is scoring a goal. Like in many other cases, the reward is sparse (there are few goals per 90 minutes) and serves as a succint representation of the task. Here I'm building a tool for testing different algorithms on a variety of environments. 

Currently, I have one algorithm for the [Acrobot-v1](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py) environment. The algorithm is an [LP formulation](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) for finding an optimal reward function from a linearly parameterized family of reward functions. 

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

The reward function is a function of the angle that top link makes with the vertical, `theta1`, and the angle the bottom link makes with respect to the top link, `theta2`.

<table>
  <tr>
   <th> True Reward Function </th>
   <th> Recovered Reward Function </th>
  <tr>
    <td> <img width="256" src="https://user-images.githubusercontent.com/7254326/118997340-36cc4b00-b9a6-11eb-9477-572415c4b647.png" /> </td>
    <td> <img width="256" src="https://user-images.githubusercontent.com/7254326/119001218-6a5ca480-b9a9-11eb-9e90-022e07d77748.png" /> </td>
  </tr>
</table>

The recovered reward function looks nothing like the true one. But importanly, the two match around the origin. The origin is the initial state, where both the links point downward. Both incentivize getting out of this region. Once the agent is out of this region, it has sufficient kinetic energy to eventually finish the task, even if no further torque is provided.

This is one trajectory under the optimal policy for the recovered reward function. The trajectory also exhibits the observed behaviour.

![acrobot](https://user-images.githubusercontent.com/7254326/119007009-80209880-b9ae-11eb-811d-e6ba0f3d169c.gif)

## Algorithm Overview 

Assuming that the observed policy is optimal for the underlying task, the algorithm iteratively shrinks the space of candidate reward function. Initially, a random reward function is guessed and an optimal policy for it is determined. Valid candidates satisfy that the value of the states under this new policy is less than or equal to that under the observed policy. This constraint shrinks the space of valid candidates. We find the *best* valid reward candidate, characterized by superiority of value of observed policy over the new policy. An optimal policy is determined for this new candidate and the cycle is repeated again. The outer loop described above can be found [here](https://github.com/Vrroom/IRL/blob/cd5f112ad4728fd12a19c09a7670ef037f6a00bc/IRL.py#L101). Reward Space shrinking can be found [here](https://github.com/Vrroom/IRL/blob/cd5f112ad4728fd12a19c09a7670ef037f6a00bc/RewardFnSpace.py#L102).
 
## Limitations

1. The main bottleneck preventing generalization to other environments is that each new environment needs to be supplied with a custom reward bases.  
2. The motivation of the problem isn't addressed in its solution. The motivation was to recover a succint representation of the task from observed behavior. For example, "score a goal" or "cross the grey line". In the example above, the recovered reward function explained the observed behaviour and yet wasn't interpretable in the same way as the true reward function.

## Future Work

1. Add Ziebart's _Maximum Entropy Inverse Reinforcement Learning_ and the more recent _Maximum Entropy Deep Inverse Reinforcement Learning_ by Wulfmeier.

## References

1. [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
2. [Linear Programming for Piecewise Linear Objectives](http://www.seas.ucla.edu/~vandenbe/ee236a/lectures/pwl.pdf)
