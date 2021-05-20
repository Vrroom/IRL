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


## Algorithm

<!-- This section requires knowledge of the MDP Planning and the Reinforcement Learning Problem. See [Sutton and Barto](http://incompleteideas.net/book/RLbook2020.pdf) if required. Also, I'll start using the word *policy* instead of *behaviour* to be consistent with the standard terminology.  -->

There are a few key components of the algorithm:

1. _Observed Behaviour_: I played the Acrobot-v1 Environment over and over again and collected trajectories. Use [Play.py](https://github.com/Vrroom/IRL/blob/master/Play.py) to create trajectories for your environment.
2. _Linearly parameterized space of reward functions_: A reward function in this space is a linear combination of a fixed set of basis functions. I chose rectangular step functions as the bases. The algorithm searches for the coefficients in the linear combination that best explain the observed behaviour.

<!-- 
We assume that the observed policy is optimal for the underlying task and then iteratively shrink the space of reward function candidates. Initially, we guess a reward function and find the optimal policy under it. For a reward function to be a valid candidate, the value of the initial state under this new policy has to be less than or equal to that under the observed policy. This constraint shrinks the space of valid candidates. We find the *best* valid reward candidate (this step requires Linear Programming to optimize the objective which defines what is best). Again, an optimal policy is determined for this new reward function and further constraints are added.

 -->
## Future Work

## References

1. [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
2. [Linear Programming for Piecewise Linear Objectives](http://www.seas.ucla.edu/~vandenbe/ee236a/lectures/pwl.pdf)
