from Network import *
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.gym import *
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

def build_and_train(run_ID=0, cuda_idx=None, n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    env = gym.make('Acrobot-v1')
    sampler = SerialSampler(
        EnvCls=GymEnvWrapper,
        env_kwargs=dict(env=env),
        batch_T=5,  # 5 time-steps per sampler iteration.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=400,
    )
    algo = A2C()  # Run with defaults.
    agent = CategoricalPgAgent(AcrobotNet)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    name = "a2c_acrobot"
    log_dir = "A2C"
    with logger_context(log_dir, run_ID, name):
        runner.train()
    torch.save(agent.state_dict(), 'Models/acrobot-a2c.pth')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
    )
